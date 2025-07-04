FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements de test
COPY tests/requirements.txt /tmp/test-requirements.txt
COPY api/requirements.txt /tmp/api-requirements.txt
COPY ml/requirements.txt /tmp/ml-requirements.txt

# Installer toutes les dépendances
RUN pip install --no-cache-dir -r /tmp/test-requirements.txt && \
    pip install --no-cache-dir -r /tmp/api-requirements.txt && \
    pip install --no-cache-dir -r /tmp/ml-requirements.txt

# Copier le code source
COPY . /app/

# Créer les dossiers de résultats
RUN mkdir -p /app/test-results

# Variables d'environnement
ENV PYTHONPATH="/app:/app/ml:/app/api:/app/webapp"
ENV TF_CPP_MIN_LOG_LEVEL=2

# Point d'entrée par défaut
CMD ["python", "-m", "pytest", "tests/", "-v"]