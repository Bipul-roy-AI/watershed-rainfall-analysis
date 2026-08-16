FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies required by geospatial libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libspatialindex-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy project metadata first to leverage Docker layer caching
COPY pyproject.toml .

# Install the package and its dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the application source
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "watershed_analyzer/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
