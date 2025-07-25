# Use official Python 3.9 image as base
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt to container
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to container
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
