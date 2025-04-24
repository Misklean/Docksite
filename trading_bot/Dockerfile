FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install TA-Lib and core dependencies via conda
RUN conda install -y -c conda-forge ta-lib numpy pandas && \
    conda clean -afy

# Copy requirements file
COPY requirements.txt .

# Install remaining Python packages with pip
# Remove packages already installed with conda
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir alpaca-py configparser joblib matplotlib neuralforecast scikit-learn tqdm torch utilsforecast xgboost psycopg2-binary

# Copy application code
COPY . .

# Run the application
CMD ["python", "main.py"]