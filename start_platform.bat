@echo off
echo Starting Honeywell Autonomous SOC Platform...

:: Start FastAPI Backend
echo Starting FastAPI Backend on port 8000...
cd backend
start "FastAPI Backend" cmd /k "python -m uvicorn api:app --host 0.0.0.0 --port 8000"
cd ..

:: Start Next.js Frontend
echo Starting Next.js Frontend on port 3000...
cd frontend
start "Next.js Frontend" cmd /k "npm run dev"
cd ..

echo Platform is running!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000
echo Please close the opened windows to stop the servers.
pause
