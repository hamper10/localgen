"""localgen -- image -> 3D and image generation on a 10 GB GPU, driven through ComfyUI.

  * image -> 3D  : TRELLIS.2 (MIT) via ComfyUI-Trellis2        -> trellis()
  * image edit   : FLUX.2 klein 4B (Apache 2.0) via core nodes -> klein_edit()
  * text -> image: FLUX.2 klein 4B                               -> klein_generate()

Talks to a running ComfyUI at http://127.0.0.1:8188 (COMFY_URL to override).
The TRELLIS.2 workflows are the UI-format examples shipped with ComfyUI-Trellis2;
they are converted to API prompts with the server's /object_info, so a node's
widget order never has to be hand-copied. COMFY_DIR points at the ComfyUI folder
so they can be found.

Every output gets a <file>.localgen.json sidecar: generator, model, seed,
settings, timing and licence -- a provenance record for the asset.
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

__version__ = "0.1.0"

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFY_DIR", os.path.join(os.getcwd(), "ComfyUI"))
TRELLIS_MODEL = os.environ.get("TRELLIS_MODEL", "visualbruno/TRELLIS.2-4B-FP8")  # fits 10 GB; "microsoft/TRELLIS.2-4B" for bf16


def _trellis_workflow(name):
    return os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-Trellis2", "example_workflows", name)


# --------------------------------------------------------------------- HTTP
def _get(path):
    with urllib.request.urlopen(COMFY_URL + path, timeout=600) as r:
        return json.loads(r.read())


def _post(path, data):
    req = urllib.request.Request(COMFY_URL + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"ComfyUI {e.code} on {path}: {body[:3000]}")


def alive():
    try:
        _get("/system_stats")
        return True
    except Exception:
        return False


def _require_server():
    if not alive():
        raise RuntimeError(f"ComfyUI not running at {COMFY_URL} (start it, or set COMFY_URL)")


def upload_image(path):
    """POST an image to ComfyUI's input folder; returns the name LoadImage wants."""
    name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(path)}"
    boundary = "----localgen" + uuid.uuid4().hex
    with open(path, "rb") as f:
        payload = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n").encode() + payload + (
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request(COMFY_URL + "/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["name"]


def run(prompt, poll=2.0, log=print):
    """Queue an API prompt, wait for it, return the history entry."""
    pid = _post("/prompt", {"prompt": prompt, "client_id": "localgen"})
    if "error" in pid:
        raise RuntimeError(json.dumps(pid, indent=1))
    pid = pid["prompt_id"]
    t0 = time.time()
    last = -1
    while True:
        h = _get(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError("ComfyUI error: " + json.dumps(msgs, indent=1)[:4000])
            log(f"[localgen] done in {time.time() - t0:.0f} s")
            return h[pid]
        n = int(time.time() - t0)
        if n // 15 != last:
            last = n // 15
            log(f"[localgen] running {n} s ...")
        time.sleep(poll)


# --------------------------------------------------------- UI -> API prompt
def ui_to_api(wf, object_info):
    """Convert a UI-format workflow to an API prompt using the server's input schemas."""
    links = {}
    for l in wf.get("links", []):
        # [id, from_node, from_slot, to_node, to_slot, type]
        links[l[0]] = (l[1], l[2])
    api = {}
    for n in wf["nodes"]:
        t = n["type"]
        if t in ("Note", "MarkdownNote", "Reroute", "PrimitiveNode"):
            continue
        info = object_info.get(t)
        if info is None:
            raise RuntimeError(f"node type {t} unknown to the server (custom node not loaded?)")
        schema = dict(info["input"].get("required", {}))
        schema.update(info["input"].get("optional", {}))
        order = list(info["input"].get("required", {}).keys()) + list(info["input"].get("optional", {}).keys())
        linked = {i["name"]: i.get("link") for i in n.get("inputs", []) if i.get("link") is not None}
        widget_names = [i["name"] for i in n.get("inputs", []) if i.get("widget")]
        wv = list(n.get("widgets_values") or [])
        inputs = {}
        wi = 0
        for name in order:
            spec = schema[name]
            kind = spec[0]
            if name in linked:
                src, slot = links[linked[name]]
                inputs[name] = [str(src), slot]
                # a linked widget input still occupies a widgets_values slot
                if name in widget_names and wi < len(wv):
                    wi += 1
                    if _has_control(name, spec):
                        wi += 1
                continue
            if isinstance(kind, list) or kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                if wi < len(wv):
                    inputs[name] = wv[wi]
                    wi += 1
                    if _has_control(name, spec):
                        wi += 1  # 'fixed' / 'randomize' after a seed
                elif len(spec) > 1 and "default" in spec[1]:
                    inputs[name] = spec[1]["default"]
        api[str(n["id"])] = {"class_type": t, "inputs": inputs}
    return api


def _has_control(name, spec):
    if len(spec) > 1 and isinstance(spec[1], dict) and spec[1].get("control_after_generate"):
        return True
    return name in ("seed", "noise_seed")


def find(api, class_type, title_in=None, wf=None):
    ids = [k for k, v in api.items() if v["class_type"] == class_type]
    if title_in and wf:
        titled = [str(n["id"]) for n in wf["nodes"] if n["type"] == class_type and title_in.lower() in (n.get("title") or "").lower()]
        if titled:
            ids = [i for i in ids if i in titled]
    return ids


def outputs_of(history):
    files = []
    for out in history.get("outputs", {}).values():
        for items in out.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and "filename" in it:
                    files.append(it)
    return files


def fetch_output(item, dst):
    sub = item.get("subfolder", "")
    typ = item.get("type", "output")
    src = os.path.join(COMFY_DIR, typ, sub, item["filename"])
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        return dst
    q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": sub, "type": typ})
    with urllib.request.urlopen(COMFY_URL + "/view?" + q, timeout=600) as r:
        with open(dst, "wb") as f:
            f.write(r.read())
    return dst


def _has_alpha(path):
    from PIL import Image
    return Image.open(path).mode in ("RGBA", "LA")


def _wire_views(api, wf, front_path, views, upload=upload_image):
    """Multi-view workflow: the example wires Front/Back loaders; add loaders for left/right."""
    cond = find(api, "Trellis2ImageCondMultiViewGenerator")
    if not cond:
        raise RuntimeError("multi-view workflow has no Trellis2ImageCondMultiViewGenerator")
    cond = cond[0]
    loaders = {}
    for n in wf["nodes"]:
        if n["type"] == "Trellis2LoadImageWithTransparency":
            t = (n.get("title") or "").lower()
            loaders["front" if "front" in t else "back" if "back" in t else t] = str(n["id"])
    nxt = [max(int(k) for k in api) + 1]

    def loader(path):
        li = str(nxt[0]); nxt[0] += 1
        api[li] = {"class_type": "Trellis2LoadImageWithTransparency", "inputs": {"image": upload(path)}}
        pp = str(nxt[0]); nxt[0] += 1
        # the example feeds the preprocessor from the loader's slot 2 (its RGBA output)
        api[pp] = {"class_type": "Trellis2PreProcessImage", "inputs": {"image": [li, 2], "padding": 10, "remove_background": not _has_alpha(path), "max_size": 1024}}
        return pp

    def set_existing(key, path):
        li = loaders[key]
        api[li]["inputs"]["image"] = upload(path)
        for node in api.values():
            if node["class_type"] == "Trellis2PreProcessImage" and (node["inputs"].get("image") or [None])[0] == li:
                node["inputs"]["remove_background"] = not _has_alpha(path)

    set_existing("front", front_path)
    if "back" in views:
        set_existing("back", views["back"])
    else:
        # no back view: drop the example's back loader and its preprocessor
        bk = loaders.get("back")
        if bk:
            for nid in [k for k, v in api.items() if v["class_type"] == "Trellis2PreProcessImage" and (v["inputs"].get("image") or [None])[0] == bk]:
                del api[nid]
            del api[bk]
            api[cond]["inputs"].pop("back_image", None)
    for side in ("left", "right"):
        if side in views:
            api[cond]["inputs"][side + "_image"] = [loader(views[side]), 0]


def panel_side_view(front_png, depth_ratio, out_png, edge_px=24):
    """Synthesise the SIDE view of a flat panel from its front concept: a vertical bar as tall as the
    panel and depth_ratio of its width wide, coloured from the panel's own edge. RGBA on transparent.
    Experimental: in testing a turned (three-quarter) concept worked better -- see the README."""
    from PIL import Image
    import numpy as np
    im = Image.open(front_png).convert("RGBA")
    w, h = im.size
    a = np.asarray(im).astype(int)
    bg = a[2, 2, :3]
    mask = (np.abs(a[:, :, :3] - bg).sum(2) > 60)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        x0, x1, y0, y1 = 0, w, 0, h
    else:
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    ph = y1 - y0
    pw = x1 - x0
    bar_w = max(8, int(round(pw * depth_ratio)))
    edge = im.crop((x1 - edge_px, y0, x1, y1)).resize((bar_w, ph))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(edge, ((w - bar_w) // 2, y0))
    out.save(out_png)
    return out_png


# ------------------------------------------------------------------ TRELLIS
def drop_cascade(api):
    """The examples chain a cascade stage (512 -> 1024); remove it and rewire its consumers.
    Its outputs (shape_slat, resolution, pipeline, num_tokens) map onto the shape generator's
    (shape_slat, resolution, pipeline) slot for slot; num_tokens becomes a literal. The multi-view
    cascade has views_list in third place: (shape_slat, resolution, views_list, pipeline, num_tokens)
    over a 4-output generator."""
    cascade = find(api, "Trellis2ShapeCascadeGenerator") + find(api, "Trellis2ShapeCascadeMultiViewGenerator")
    for cid in cascade:
        src_shape = api[cid]["inputs"].get("shape_slat")  # [shape_gen_id, 0]
        sg = src_shape[0] if isinstance(src_shape, list) else None
        tokens = api[cid]["inputs"].get("max_num_tokens", 999999)
        passthrough = 4 if "MultiView" in api[cid]["class_type"] else 3
        for node in api.values():
            for k, v in list(node["inputs"].items()):
                if isinstance(v, list) and v and v[0] == cid:
                    slot = v[1]
                    if slot < passthrough and sg is not None:
                        node["inputs"][k] = [sg, slot]
                    else:
                        node["inputs"][k] = tokens
        del api[cid]


def tune_trellis(api, res="512", seed=12345, faces=200000, texture_size=1024, tex_res=512, prefix="localgen"):
    """Set the 10 GB-safe settings on a converted TRELLIS.2 API prompt (in place)."""
    for nid in find(api, "Trellis2LoadModel"):
        api[nid]["inputs"]["modelname"] = TRELLIS_MODEL
        api[nid]["inputs"]["low_vram"] = True
        api[nid]["inputs"]["backend"] = os.environ.get("TRELLIS_ATTN", "sdpa")
        api[nid]["inputs"]["sparse_backend"] = os.environ.get("TRELLIS_SPARSE_ATTN", "xformers")
        api[nid]["inputs"]["keep_models_loaded"] = False
    for nid in find(api, "Trellis2SparseGenerator") + find(api, "Trellis2MeshTexturing") + find(api, "Trellis2SparseMultiViewGenerator"):
        api[nid]["inputs"]["seed"] = seed
    for nid in find(api, "Trellis2UnWrapAndRasterizer"):
        if not isinstance(api[nid]["inputs"].get("texture_size"), list):
            api[nid]["inputs"]["texture_size"] = int(texture_size)
    # texturing at 1024 cond + 2048 texture OOMs on 10 GB; 512 + 1024 is the default here
    for nid in find(api, "Trellis2MeshTexturing"):
        api[nid]["inputs"]["resolution"] = int(tex_res)
        if not isinstance(api[nid]["inputs"].get("texture_size"), list):
            api[nid]["inputs"]["texture_size"] = int(texture_size)
    shape_res = 512 if str(res).startswith("512") else 1024
    for nid in find(api, "Trellis2ShapeGenerator") + find(api, "Trellis2ShapeMultiViewGenerator") + find(api, "Trellis2TexSlatMultiViewGenerator"):
        api[nid]["inputs"]["resolution"] = shape_res
    if not str(res).endswith("cascade"):
        drop_cascade(api)
    # the quad reconstruction at 1024 over a 512 shape OOMs on 10 GB (and the Windows driver spills
    # to system RAM first, so the run crawls before it fails); keep it at the shape's resolution
    for nid in find(api, "Trellis2ReconstructMeshWithQuad"):
        api[nid]["inputs"]["resolution"] = shape_res
    for nid in find(api, "Trellis2SimplifyMesh"):
        if not isinstance(api[nid]["inputs"].get("target_face_num"), list):
            api[nid]["inputs"]["target_face_num"] = faces
    for nid in find(api, "PrimitiveInt"):
        if api[nid]["inputs"].get("value") in (500000, 300000):
            api[nid]["inputs"]["value"] = faces
        if api[nid]["inputs"].get("value") == 2048:
            api[nid]["inputs"]["value"] = texture_size
    for nid in find(api, "Trellis2ExportMesh"):
        api[nid]["inputs"]["file_format"] = "glb"
    for nid in find(api, "PrimitiveString"):
        api[nid]["inputs"]["value"] = prefix
    for nid in find(api, "Preview3D"):
        del api[nid]
    return api


def trellis(image_path, out_glb, res="512", seed=12345, faces=200000, texture=True, texture_size=1024, tex_res=512,
            views=None, log=print):
    """Image -> textured GLB with TRELLIS.2. res: 512 | 1024 | 1024_cascade | 1536_cascade.
    views: optional extra views {"back": png, "left": png, "right": png}.
    Returns a provenance dict (also written next to the GLB)."""
    _require_server()
    oi = _get("/object_info")
    views = dict(views or {})
    mv = bool(views)
    wf_name = "MeshWithTexturing_MultiView.json" if mv else ("MeshWithTexturing.json" if texture else "MeshOnly.json")
    wf_path = _trellis_workflow(wf_name)
    if not os.path.exists(wf_path):
        raise RuntimeError(f"{wf_path} not found: set COMFY_DIR to your ComfyUI folder (with ComfyUI-Trellis2 installed)")
    with open(wf_path, encoding="utf-8") as f:
        wf = json.load(f)
    api = ui_to_api(wf, oi)
    prefix = "localgen_" + uuid.uuid4().hex[:8]
    if mv:
        _wire_views(api, wf, image_path, views)
    else:
        name = upload_image(image_path)
        for nid in find(api, "Trellis2LoadImageWithTransparency"):
            api[nid]["inputs"]["image"] = name
        # the preprocessor wants RGBA: cut the background (rembg) unless the PNG already has alpha
        for nid in find(api, "Trellis2PreProcessImage"):
            api[nid]["inputs"]["remove_background"] = not _has_alpha(image_path)
    tune_trellis(api, res=res, seed=seed, faces=faces, texture_size=texture_size, tex_res=tex_res, prefix=prefix)
    t0 = time.time()
    hist = run(api, log=log)
    outs = [o for o in outputs_of(hist) if o["filename"].lower().endswith(".glb")]
    if not outs:
        raise RuntimeError("no GLB in outputs: " + json.dumps(hist.get("outputs"))[:1000])
    # prefer the textured one
    outs.sort(key=lambda o: ("Textured" not in o["filename"], o["filename"]))
    fetch_output(outs[0], out_glb)
    prov = {
        "generator": "localgen TRELLIS.2 (ComfyUI-Trellis2)",
        "localgen_version": __version__,
        "model": TRELLIS_MODEL,
        "resolution": str(res),
        "seed": seed,
        "faces": faces,
        "textured": texture,
        "views": {k: os.path.abspath(v) for k, v in views.items()} if views else None,
        "seconds": round(time.time() - t0, 1),
        "license": "MIT (TRELLIS.2 weights + code); DINOv3 encoder under the DINOv3 License",
        "source_image": os.path.abspath(image_path),
    }
    with open(out_glb + ".localgen.json", "w") as f:
        json.dump(prov, f, indent=1)
    log(f"[localgen] {out_glb} ({os.path.getsize(out_glb) / 1e6:.1f} MB)")
    return prov


# ------------------------------------------------------------- FLUX.2 klein
KLEIN_UNET = {"distilled": "flux-2-klein-4b-fp8.safetensors", "base": "flux-2-klein-base-4b-fp8.safetensors"}
KLEIN_CLIP = "qwen_3_4b.safetensors"
KLEIN_VAE = "flux2-vae.safetensors"


def klein_graph(prompt, images, w, h, base, seed, negative=""):
    """Build the FLUX.2 klein 4B API prompt from core nodes (the shipped templates are subgraphs).
    Distilled: 4 steps, cfg 1, zeroed negative. Base: 20 steps, cfg 5, text negative."""
    g = {}
    nid = [0]

    def add(class_type, **inputs):
        nid[0] += 1
        g[str(nid[0])] = {"class_type": class_type, "inputs": inputs}
        return str(nid[0])

    unet = add("UNETLoader", unet_name=KLEIN_UNET["base" if base else "distilled"], weight_dtype="default")
    clip = add("CLIPLoader", clip_name=KLEIN_CLIP, type="flux2", device="default")
    vae = add("VAELoader", vae_name=KLEIN_VAE)
    pos = add("CLIPTextEncode", clip=[clip, 0], text=prompt)
    if base:
        neg = add("CLIPTextEncode", clip=[clip, 0], text=negative)
    else:
        neg = add("ConditioningZeroOut", conditioning=[pos, 0])
    if images:
        first = None
        for img in images:
            li = add("LoadImage", image=img)
            sc = add("ImageScaleToTotalPixels", image=[li, 0], upscale_method="nearest-exact", megapixels=1.0, resolution_steps=1)
            if first is None:
                first = sc
            enc = add("VAEEncode", pixels=[sc, 0], vae=[vae, 0])
            pos = add("ReferenceLatent", conditioning=[pos, 0], latent=[enc, 0])
            neg = add("ReferenceLatent", conditioning=[neg, 0], latent=[enc, 0])
        size = add("GetImageSize", image=[first, 0])
        wh = {"width": [size, 0], "height": [size, 1]}
    else:
        wh = {"width": w, "height": h}
    latent = add("EmptyFlux2LatentImage", batch_size=1, **wh)
    sigmas = add("Flux2Scheduler", steps=20 if base else 4, **wh)
    guider = add("CFGGuider", model=[unet, 0], positive=[pos, 0], negative=[neg, 0], cfg=5.0 if base else 1.0)
    noise = add("RandomNoise", noise_seed=seed)
    sampler = add("KSamplerSelect", sampler_name="euler")
    samp = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=[sampler, 0], sigmas=[sigmas, 0], latent_image=[latent, 0])
    dec = add("VAEDecode", samples=[samp, 0], vae=[vae, 0])
    add("SaveImage", images=[dec, 0], filename_prefix="localgen/klein")
    return g


def _seed(seed):
    return int.from_bytes(os.urandom(4), "little") if seed is None else seed


def _run_klein(g, out_png, prov, log):
    t0 = time.time()
    hist = run(g, log=log)
    outs = [o for o in outputs_of(hist) if o["filename"].lower().endswith(".png") and o.get("type") == "output"]
    if not outs:
        raise RuntimeError("no PNG in outputs: " + json.dumps(hist.get("outputs"))[:1000])
    fetch_output(outs[0], out_png)
    prov["localgen_version"] = __version__
    prov["seconds"] = round(time.time() - t0, 1)
    prov["license"] = "Apache 2.0 (FLUX.2 klein 4B)"
    with open(out_png + ".localgen.json", "w") as f:
        json.dump(prov, f, indent=1)
    log(f"[localgen] {out_png}")
    return prov


def klein_edit(images, prompt, out_png, base=False, seed=None, negative="", log=print):
    """Edit/restyle. images[0] is the subject (sets the output size), the rest are references
    the prompt can name as image 2, image 3 ... Output follows the first image's size (1 MP)."""
    _require_server()
    seed = _seed(seed)
    names = [upload_image(p) for p in images]
    g = klein_graph(prompt, names, 1024, 1024, base, seed, negative)
    prov = {"generator": "localgen FLUX.2 klein 4B " + ("base" if base else "distilled"), "mode": "edit", "prompt": prompt,
            "negative": negative, "inputs": [os.path.abspath(p) for p in images], "seed": seed}
    return _run_klein(g, out_png, prov, log)


def klein_generate(prompt, out_png, w=1024, h=1024, base=False, seed=None, negative="", log=print):
    _require_server()
    seed = _seed(seed)
    g = klein_graph(prompt, [], w, h, base, seed, negative)
    prov = {"generator": "localgen FLUX.2 klein 4B " + ("base" if base else "distilled"), "mode": "generate", "prompt": prompt,
            "negative": negative, "size": [w, h], "seed": seed}
    return _run_klein(g, out_png, prov, log)


# ---------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(prog="localgen", description="Image -> 3D (TRELLIS.2) and image generation (FLUX.2 klein 4B) on a 10 GB GPU via ComfyUI.")
    p.add_argument("--version", action="version", version=f"localgen {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trellis", help="image -> textured GLB")
    t.add_argument("image")
    t.add_argument("out")
    t.add_argument("--res", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    t.add_argument("--seed", type=int, default=12345)
    t.add_argument("--faces", type=int, default=200000, help="target face count after simplification")
    t.add_argument("--no-texture", action="store_true", help="mesh only")
    t.add_argument("--tex", type=int, default=1024, help="texture size in px")
    t.add_argument("--texres", type=int, default=512, help="texturing conditioning resolution")
    t.add_argument("--back", help="extra view: back")
    t.add_argument("--left", help="extra view: left")
    t.add_argument("--right", help="extra view: right")
    t.add_argument("--side", type=float, help="experimental: synthesise a side view for a flat panel, depth/width ratio")

    e = sub.add_parser("edit", help="edit/restyle images with FLUX.2 klein")
    e.add_argument("out")
    e.add_argument("prompt")
    e.add_argument("images", nargs="+", help="subject first, then reference images")
    e.add_argument("--base", action="store_true", help="20-step cfg-5 base model (slower, steerable); default is the 4-step distilled one")
    e.add_argument("--seed", type=int)
    e.add_argument("--neg", default="", help="negative prompt (base model only)")

    g = sub.add_parser("gen", help="text -> image with FLUX.2 klein")
    g.add_argument("out")
    g.add_argument("prompt")
    g.add_argument("--w", type=int, default=1024)
    g.add_argument("--h", type=int, default=1024)
    g.add_argument("--base", action="store_true")
    g.add_argument("--seed", type=int)
    g.add_argument("--neg", default="")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "trellis":
            views = {k: getattr(a, k) for k in ("back", "left", "right") if getattr(a, k)}
            if a.side is not None:
                views["left"] = panel_side_view(a.image, a.side, a.out + ".side.png")
            trellis(a.image, a.out, res=a.res, seed=a.seed, faces=a.faces, texture=not a.no_texture,
                    texture_size=a.tex, tex_res=a.texres, views=views or None)
        elif a.cmd == "edit":
            klein_edit(a.images, a.prompt, a.out, base=a.base, seed=a.seed, negative=a.neg)
        elif a.cmd == "gen":
            klein_generate(a.prompt, a.out, w=a.w, h=a.h, base=a.base, seed=a.seed, negative=a.neg)
    except RuntimeError as err:
        print(f"localgen: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
