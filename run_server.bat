@echo off
rem d3-mini System One HTTP server, localhost only. Port 8000 is taken by Docker on this PC, so 8899.
rem Test: curl -X POST localhost:8899/v1/systemone -H "Content-Type: application/json" --data-binary @req.json
cd /d %~dp0
set PYTHONIOENCODING=utf-8
set SNAP=%USERPROFILE%\.cache\huggingface\hub\models--vllm-sr--d3-mini\snapshots\6d8971f464d6f9d4c5af7d9a8d75f3aa03be4f59
.venv\Scripts\python.exe "%SNAP%\d3_server.py" --port 8899 %*
