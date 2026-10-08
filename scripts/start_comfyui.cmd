@echo off
rem Start ComfyUI for localgen. Run from your ComfyUI folder's parent, or set COMFY_DIR.
rem Usage: start_comfyui.cmd [extra ComfyUI args]
if "%COMFY_DIR%"=="" set COMFY_DIR=%CD%\ComfyUI
cd /d "%COMFY_DIR%"
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if exist ..\venv\Scripts\python.exe (set PY=..\venv\Scripts\python.exe) else (set PY=python)
%PY% main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --lowvram %*
