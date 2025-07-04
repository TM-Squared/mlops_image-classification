# 🌱 Plant Classification MLOps

**Classification d'images de plantes (Pissenlit vs Herbe) avec pipeline MLOps complet**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.10.1-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.7.1-green)
![MinIO](https://img.shields.io/badge/MinIO-S3-yellow)

## 📋 Table des Matières

- [Aperçu du Projet](#aperçu-du-projet)
- [Architecture](#architecture)
- [Technologies Utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API Documentation](#api-documentation)
- [Tests](#tests)
- [Monitoring](#monitoring)
- [Déploiement](#déploiement)
- [Contribution](#contribution)

## Aperçu du Projet

Ce projet implémente un pipeline MLOps complet pour la classification binaire d'images de plantes, distinguant les dandelion de l'herbe (grass). Il démontre les meilleures pratiques MLOps incluant l'automatisation, le monitoring, et le déploiement continu.

### Fonctionnalités Principales

- **🤖 Classification automatique** d'images avec TensorFlow/MobileNetV2
- **📊 Pipeline d'entraînement** automatisé avec Apache Airflow
- **🗄️ Stockage distribué** avec MinIO (compatible S3)
- **📈 Tracking d'expériences** avec MLflow
- **🌐 API REST** avec FastAPI
- **💻 Interface web** avec Streamlit
- **🔄 Entraînement continu** et déploiement automatique
- **🐳 Containerisation** complète avec Docker

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Data Storage  │    │   Processing    │
│                 │    │                 │    │                 │
│ • GitHub URLs   │───▶│ • MinIO (S3)    │───▶│ • Apache Airflow│
│ • Manual Upload │    │ • MySQL         │    │ • TensorFlow    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Model Store   │    │   ML Training   │
│                 │    │                 │    │                 │
│ • MLflow UI     │◀───│ • MinIO Models  │◀───│ • Model Training│
│ • Logs          │    │ • Model Registry│    │ • Evaluation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Interface│    │   API Layer     │    │   Deployment    │
│                 │    │                 │    │                 │
│ • Streamlit App │───▶│ • FastAPI       │◀───│ • Auto Deploy   │
│ • Web Interface │    │ • REST Endpoints│    │ • Model Serving │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Technologies Utilisées

### Machine Learning & Data
- **TensorFlow 2.13** - Framework de deep learning
- **MobileNetV2** - Modèle de transfer learning léger
- **MLflow** - Tracking d'expériences et registry de modèles
- **Pandas/NumPy** - Manipulation de données

### Infrastructure & Orchestration
- **Apache Airflow** - Orchestration de pipelines
- **Docker & Docker Compose** - Containerisation
- **MinIO** - Stockage objet compatible S3
- **MySQL** - Base de données relationnelle
- **PostgreSQL** - Base de données Airflow

### API & Interface
- **FastAPI** - API REST moderne et rapide
- **Streamlit** - Interface web interactive
- **Uvicorn** - Serveur ASGI haute performance

### DevOps & Monitoring
- **GitHub Actions** - CI/CD (prêt pour déploiement)
- **pytest** - Framework de tests
- **Logging** - Monitoring et debugging

## 🚀 Installation

### Prérequis

- Docker & Docker Compose
- Git
- 8GB RAM minimum
- 10GB espace disque libre

### Installation Rapide

```bash
# 1. Cloner le repository
git clone <votre-repo-url>
cd plant-classification-mlops

# 2. Créer le fichier d'environnement
cp .env.example .env

# 3. Créer les dossiers nécessaires
mkdir -p airflow/logs models tests/data

# 4. Lancer l'environnement
docker-compose up --build -d

# 5. Attendre le démarrage (2-3 minutes)
docker-compose logs -f
```

### Configuration des Variables d'Environnement

Modifier le fichier `.env` selon vos besoins :

```bash
# Base de données
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow123
MYSQL_USER=plants_user
MYSQL_PASSWORD=plants123

# MinIO
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Airflow
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin123
```

## Utilisation

### 1. Premier Démarrage

Après le lancement, configurez les connexions Airflow :

```bash
# Accéder à Airflow
open http://localhost:8080
# Login: admin / admin123

# Déclencher le DAG "setup_connections" pour configurer automatiquement les connexions
```

### 2. Pipeline d'Ingestion de Données

```bash
# Dans Airflow UI, activer et déclencher :
# 1. "plants_data_ingestion_pipeline" - Ingestion des données
# 2. "model_training_minio_pipeline" - Entraînement du modèle
```

### 3. Test de l'API

```bash
# Vérifier l'API
curl http://localhost:8000/health

# Tester une prédiction
curl -X POST "http://localhost:8000/predict-url" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg"}'
```

### 4. Interface Web

```bash
# Accéder à la WebApp
open http://localhost:8501
```

## 🔌 API Documentation

### Endpoints Principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Informations sur l'API |
| `/health` | GET | Statut de santé |
| `/predict` | POST | Prédiction via upload |
| `/predict-url` | POST | Prédiction via URL |
| `/models` | GET | Liste des modèles |
| `/reload-model` | POST | Recharger le modèle |

### Exemple d'Utilisation

```python
import requests

# Prédiction via URL
response = requests.post(
    "http://localhost:8000/predict-url",
    params={"image_url": "https://example.com/image.jpg"}
)

result = response.json()
print(f"Classe prédite: {result['predicted_class']}")
print(f"Confiance: {result['confidence']:.2%}")
```

### Documentation Interactive

La documentation Swagger est disponible à : `http://localhost:8000/docs`

## Tests

### Lancer les Tests

```bash
# Tests unitaires
docker-compose exec airflow-webserver python -m pytest tests/ -v

# Tests d'intégration
docker-compose exec api python -m pytest tests/ -v

# Tests end-to-end
python tests/test_e2e.py
```

### Coverage

```bash
# Générer un rapport de couverture
docker-compose exec airflow-webserver python -m pytest tests/ --cov=ml --cov-report=html
```

## 📊 Monitoring

### Interfaces de Monitoring

- **Airflow** : `http://localhost:8080` - Monitoring des DAGs
- **MLflow** : `http://localhost:5000` - Tracking des expériences
- **MinIO Console** : `http://localhost:9001` - Gestion du stockage
- **API Docs** : `http://localhost:8000/docs` - Documentation API

### Logs

```bash
# Logs en temps réel
docker-compose logs -f

# Logs spécifiques
docker-compose logs airflow-scheduler
docker-compose logs api
```

### Métriques Importantes

- **Précision du modèle** : Suivi dans MLflow
- **Temps de réponse API** : Logs FastAPI
- **Utilisation stockage** : Console MinIO
- **Statut des DAGs** : Interface Airflow

## 🚢 Déploiement

### Environnement de Production

```bash
# Utiliser le fichier de production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Ou déployment Kubernetes (manifests dans k8s/)
kubectl apply -f k8s/
```

### CI/CD avec GitHub Actions

Le pipeline CI/CD est configuré dans `.github/workflows/` et inclut :

- Tests automatiques
- Build et push des images Docker
- Déploiement automatique
- Tests de santé post-déploiement

### Variables de Production

```bash
# Production .env
POSTGRES_PASSWORD=<strong-password>
MYSQL_PASSWORD=<strong-password>
MINIO_SECRET_KEY=<strong-secret>
AIRFLOW_PASSWORD=<strong-password>
```

## 🔧 Dépannage

### Problèmes Courants

**1. Erreur de permissions MinIO**
```bash
# Vérifier les clés d'accès
docker-compose logs minio
# Recréer les connexions Airflow
```

**2. Modèle non trouvé**
```bash
# Vérifier les modèles dans MinIO
curl http://localhost:8000/models
# Relancer l'entraînement
```

**3. Erreur de base de données**
```bash
# Réinitialiser les bases
docker-compose down -v
docker-compose up --build -d
```

### Support

Pour obtenir de l'aide :
1. Vérifiez les [issues GitHub](../../issues)
2. Consultez les logs : `docker-compose logs`
3. Ouvrez une nouvelle issue avec les détails

## 👥 Contribution

### Guide de Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Standards de Code

- **Format** : Black pour Python
- **Linting** : Flake8
- **Tests** : pytest avec coverage > 80%
- **Documentation** : Docstrings pour toutes les fonctions

### Structure des Commits

```
type(scope): description

- feat: nouvelle fonctionnalité
- fix: correction de bug
- docs: documentation
- test: ajout de tests
- refactor: refactoring
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.


