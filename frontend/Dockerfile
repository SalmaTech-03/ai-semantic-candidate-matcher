# Step 1: Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY frontend/ .
# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
