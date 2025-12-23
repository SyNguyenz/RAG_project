@echo off
echo Starting Image Retrieval API...

REM Activate virtual environment nếu có
REM call venv\Scripts\activate

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%cd%

REM Start API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload