FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables (overridable at runtime)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-3.5-turbo"
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1

# Expose port for HF Spaces (FastAPI server)
EXPOSE 7860

# Default command: run FastAPI server for HF Spaces
CMD ["uvicorn", "app_server:app", "--host", "0.0.0.0", "--port", "7860"]
