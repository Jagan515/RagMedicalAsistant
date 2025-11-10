FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install dependencies
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


COPY . /app

# Expose Render-assigned port
EXPOSE $PORT

# Start app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "3"]