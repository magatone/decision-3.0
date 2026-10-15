@echo off
rem d3 test GUI -> http://127.0.0.1:7860 (localhost only). Other model: run_gui.bat --model vllm-sr/d3-nano
cd /d %~dp0
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe gui.py %*
