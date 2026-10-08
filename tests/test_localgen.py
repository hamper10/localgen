"""Offline tests: workflow conversion and prompt building, no GPU or ComfyUI server needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import localgen  # noqa: E402

OBJECT_INFO = {
    "Loader": {"input": {"required": {"image": [["a.png", "b.png"]]}}},
    "Sampler": {"input": {
        "required": {"model": ["MODEL"], "seed": ["INT", {"default": 0, "control_after_generate": True}], "steps": ["INT", {"default": 20}]},
        "optional": {"cfg": ["FLOAT", {"default": 7.5}]},
    }},
}


def test_ui_to_api_links_widgets_and_seed_control():
    wf = {
        "nodes": [
            {"id": 1, "type": "Loader", "widgets_values": ["b.png"]},
            {"id": 2, "type": "Sampler", "inputs": [{"name": "model", "link": 10}], "widgets_values": [42, "randomize", 8]},
            {"id": 3, "type": "Note", "widgets_values": ["ignored"]},
        ],
        "links": [[10, 1, 0, 2, 0, "MODEL"]],
    }
    api = localgen.ui_to_api(wf, OBJECT_INFO)
    assert api["1"] == {"class_type": "Loader", "inputs": {"image": "b.png"}}
    # 'randomize' after the seed is skipped; cfg falls back to its default
    assert api["2"]["inputs"] == {"model": ["1", 0], "seed": 42, "steps": 8, "cfg": 7.5}
    assert "3" not in api


def test_ui_to_api_linked_widget_keeps_its_slot():
    wf = {
        "nodes": [
            {"id": 1, "type": "Loader", "widgets_values": ["a.png"]},
            {"id": 2, "type": "Sampler",
             "inputs": [{"name": "model", "link": 10}, {"name": "seed", "link": 11, "widget": {"name": "seed"}}],
             "widgets_values": [0, "fixed", 12]},
        ],
        "links": [[10, 1, 0, 2, 0, "MODEL"], [11, 1, 0, 2, 1, "INT"]],
    }
    api = localgen.ui_to_api(wf, OBJECT_INFO)
    assert api["2"]["inputs"]["seed"] == ["1", 0]
    assert api["2"]["inputs"]["steps"] == 12


def test_ui_to_api_unknown_node_says_which():
    with pytest.raises(RuntimeError, match="Mystery"):
        localgen.ui_to_api({"nodes": [{"id": 1, "type": "Mystery"}]}, OBJECT_INFO)


def test_drop_cascade_rewires_consumers():
    api = {
        "1": {"class_type": "Trellis2ShapeGenerator", "inputs": {}},
        "2": {"class_type": "Trellis2ShapeCascadeGenerator", "inputs": {"shape_slat": ["1", 0], "max_num_tokens": 4096}},
        "3": {"class_type": "Trellis2Decode", "inputs": {"shape_slat": ["2", 0], "resolution": ["2", 1], "pipeline": ["2", 2], "tokens": ["2", 3]}},
    }
    localgen.drop_cascade(api)
    assert "2" not in api
    assert api["3"]["inputs"] == {"shape_slat": ["1", 0], "resolution": ["1", 1], "pipeline": ["1", 2], "tokens": 4096}


def test_tune_trellis_10gb_defaults():
    api = {
        "1": {"class_type": "Trellis2LoadModel", "inputs": {}},
        "2": {"class_type": "Trellis2ShapeGenerator", "inputs": {"resolution": 1024}},
        "3": {"class_type": "Trellis2ReconstructMeshWithQuad", "inputs": {"resolution": 1024}},
        "4": {"class_type": "Trellis2MeshTexturing", "inputs": {"resolution": 1024, "texture_size": 2048, "seed": 0}},
        "5": {"class_type": "Preview3D", "inputs": {}},
    }
    localgen.tune_trellis(api, res="512", seed=7)
    assert api["1"]["inputs"]["low_vram"] is True
    assert api["2"]["inputs"]["resolution"] == 512
    assert api["3"]["inputs"]["resolution"] == 512
    assert api["4"]["inputs"] == {"resolution": 512, "texture_size": 1024, "seed": 7}
    assert "5" not in api


@pytest.mark.parametrize("base,steps,cfg,neg_type", [(False, 4, 1.0, "ConditioningZeroOut"), (True, 20, 5.0, "CLIPTextEncode")])
def test_klein_graph_modes(base, steps, cfg, neg_type):
    g = localgen.klein_graph("a castle", [], 512, 768, base, seed=1)
    by_type = {v["class_type"]: v for v in g.values()}
    assert by_type["Flux2Scheduler"]["inputs"]["steps"] == steps
    assert by_type["CFGGuider"]["inputs"]["cfg"] == cfg
    neg_id = by_type["CFGGuider"]["inputs"]["negative"][0]
    assert g[neg_id]["class_type"] == neg_type
    assert by_type["EmptyFlux2LatentImage"]["inputs"]["width"] == 512


def test_klein_graph_edit_chains_references():
    g = localgen.klein_graph("restyle", ["s.png", "ref.png"], 1024, 1024, False, seed=1)
    types = [v["class_type"] for v in g.values()]
    assert types.count("LoadImage") == 2
    assert types.count("ReferenceLatent") == 4  # positive and negative, per image
    latent = next(v for v in g.values() if v["class_type"] == "EmptyFlux2LatentImage")
    assert isinstance(latent["inputs"]["width"], list)  # size follows the first image


def test_cli_parses_trellis():
    a = localgen.build_parser().parse_args(["trellis", "in.png", "out.glb", "--res", "1024", "--left", "l.png"])
    assert (a.image, a.out, a.res, a.left, a.faces) == ("in.png", "out.glb", "1024", "l.png", 200000)
