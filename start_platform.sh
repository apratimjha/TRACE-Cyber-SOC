#!/bin/bash
# Platform Startup Script

echo "Starting dasboard"

# Ensure we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Start the FastAPI Backend
echo "Starting FastAPI Backend on port 8000..."
cd backend
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Start the Next.js Frontend
echo "Starting Next.js Frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Platform is running!"
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:3000"
echo "Press Ctrl+C to stop both servers."

# Wait for both processes
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
