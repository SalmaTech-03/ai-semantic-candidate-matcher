#!/bin/bash
# Start the backend Flask server in the background
echo "Starting backend server..."
cd backend/

gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:5001 app:app &

# Go back to the root and start the frontend Streamlit app
echo "Starting frontend server..."
cd ../frontend/
streamlit run app.py --server.port $PORT --server.address=0.0.0.0