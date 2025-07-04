# Projet MLOps : Classification d'Images - Pissenlits vs Herbe

## 🎯 Objectif du Projet

Ce projet implémente un pipeline MLOps complet pour la classification binaire d'images, distinguant les pissenlits (dandelion) de l'herbe (grass). Il démontre l'application pratique des principes MLOps avec TensorFlow, incluant l'entraînement automatisé, le déploiement, le monitoring et l'intégration continue.

## 📋 Table des Matières

- [Architecture du Projet](#architecture-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Environnements](#environnements)
- [Pipeline MLOps](#pipeline-mlops)
- [API et WebApp](#api-et-webapp)
- [Monitoring](#monitoring)
- [Tests](#tests)
- [Déploiement](#déploiement)
- [Contributions](#contributions)

## 🏗️ Architecture du Projet

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│   Apache        │───▶│   ML Model      │
│   (MySQL DB)    │    │   Airflow       │    │   Training      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   CI/CD         │    │   Model         │
│   (Prometheus)  │    │   (GitHub       │    │   Registry      │
│                 │    │   Actions)      │    │   (S3/MLflow)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebApp        │    │   Kubernetes    │    │   FastAPI       │
│   (Streamlit)   │    │   Deployment    │    │   Serving       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Prérequis

- **Python 3.8+**
- **Docker & Docker Compose**
- **Kubernetes** (Minikube ou Docker Desktop)
- **Git**

### Technologies Utilisées

| Composant | Technologie |
|-----------|-------------|
| **ML Framework** | TensorFlow 2.x |
| **Orchestration** | Apache Airflow |
| **API** | FastAPI |
| **WebApp** | Streamlit |
| **Model Registry** | MLflow |
| **Storage** | AWS S3 (Minio) |
| **Database** | MySQL |
| **Monitoring** | Prometheus + Grafana |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |
| **Deployment** | Kubernetes |

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/TM-Squared/mlops_image-classification.git
cd mlops_image-classification
```

### 2. Environnement de développement

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configuration avec Docker Compose

```bash
# Lancer l'environnement complet
docker-compose up -d

# Vérifier les services
docker-compose ps
```

## 📁 Structure du Projet

```
mlops_image-classification/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Chargement des données
│   │   └── preprocessing.py    # Préprocessing des images
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py           # Définition du modèle
│   │   └── training.py        # Entraînement
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI application
│   │   └── schemas.py        # Modèles Pydantic
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # Configuration
│       └── logger.py         # Logging
├── airflow/
│   ├── dags/
│   │   ├── data_pipeline.py  # Pipeline d'extraction
│   │   ├── training_pipeline.py  # Pipeline d'entraînement
│   │   └── inference_pipeline.py # Pipeline d'inférence
│   └── plugins/
├── webapp/
│   ├── app.py               # Application Streamlit
│   └── utils.py
├── tests/
│   ├── unit/
│   │   ├── test_data_loader.py
│   │   ├── test_model.py
│   │   └── test_api.py
│   └── integration/
│       └── test_pipeline.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.webapp
│   └── Dockerfile.airflow
├── k8s/
│   ├── namespace.yaml
│   ├── api-deployment.yaml
│   ├── webapp-deployment.yaml
│   └── monitoring/
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── tests.yml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🎮 Utilisation

### 1. Préparation des données

```bash
# Initialiser la base de données
python src/data/init_db.py

# Lancer le pipeline d'extraction
python src/data/data_loader.py
```

### 2. Entraînement du modèle

```bash
# Entraînement local
python src/models/training.py

# Ou via Airflow
# Accéder à http://localhost:8080
# Déclencher le DAG "training_pipeline"
```

### 3. Lancement de l'API

```bash
# Mode développement
uvicorn src.api.main:app --reload --port 8000

# Accéder à la documentation : http://localhost:8000/docs
```

### 4. WebApp Streamlit

```bash
# Lancer l'application
streamlit run webapp/app.py

# Accéder à : http://localhost:8501
```

## 🌍 Environnements

### Développement (Local)
- **Services** : Docker Compose
- **Base de données** : MySQL local
- **Storage** : MinIO local
- **Monitoring** : Prometheus + Grafana locaux

### Production (Kubernetes)
- **Orchestration** : Kubernetes (Minikube/Cloud)
- **Base de données** : MySQL avec persistance
- **Storage** : S3 compatible
- **Monitoring** : Stack Prometheus complet
- **Ingress** : Nginx Ingress Controller

## 🔄 Pipeline MLOps

### 1. Pipeline de Données (`data_pipeline.py`)
- Extraction des images depuis les URLs
- Validation et nettoyage des données
- Stockage dans S3
- Mise à jour des métadonnées

### 2. Pipeline d'Entraînement (`training_pipeline.py`)
- Chargement des données depuis S3
- Préprocessing et augmentation
- Entraînement du modèle TensorFlow
- Évaluation et validation
- Sauvegarde du modèle dans MLflow

### 3. Pipeline de Déploiement (`inference_pipeline.py`)
- Téléchargement du meilleur modèle
- Déploiement automatique de l'API
- Tests de santé du service
- Monitoring des performances

### 4. Entraînement Continu (CT)
- **Triggers** :
  - Nouvelles données disponibles
  - Entraînement hebdomadaire programmé
  - Dégradation des performances détectée
- **Actions** :
  - Ré-entraînement automatique
  - Validation A/B testing
  - Déploiement conditionnel

## 🚦 API et WebApp

### API FastAPI

**Endpoints principaux :**
- `POST /predict` : Prédiction sur une image
- `GET /health` : Santé du service  
- `GET /metrics` : Métriques Prometheus
- `GET /model/info` : Informations sur le modèle

**Exemple d'utilisation :**
```python
import requests

# Prédiction
files = {"file": open("image.jpg", "rb")}
response = requests.post("http://localhost:8000/predict", files=files)
print(response.json())
```

### WebApp Streamlit

**Fonctionnalités :**
- Interface de téléchargement d'images
- Affichage des prédictions en temps réel
- Visualisation des métriques du modèle
- Historique des prédictions

## 📊 Monitoring

### Métriques Collectées

1. **Métriques Applicatives**
   - Nombre de prédictions
   - Latence des requêtes
   - Accuracy du modèle
   - Distributions des prédictions

2. **Métriques Système**
   - Utilisation CPU/RAM
   - Espace disque
   - Statut des services

3. **Métriques Business**
   - Taux d'utilisation
   - Temps de réponse utilisateur
   - Disponibilité du service

### Dashboards Grafana

- **Dashboard Principal** : Vue d'ensemble des services
- **Dashboard ML** : Métriques spécifiques au modèle
- **Dashboard Infrastructure** : Monitoring système

### Alertes

- Service indisponible
- Dégradation des performances
- Erreurs d'entraînement
- Espace disque faible

## 🧪 Tests

### Tests Unitaires

```bash
# Lancer tous les tests
pytest tests/unit/

# Tests spécifiques
pytest tests/unit/test_model.py -v
```

### Tests d'Intégration

```bash
# Tests end-to-end
pytest tests/integration/

# Tests avec couverture
pytest --cov=src tests/
```

### Tests de Charge

```bash
# Installer Locust
pip install locust

# Lancer les tests
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 🚀 Déploiement

### Déploiement Local (Kubernetes)

```bash
# Démarrer Minikube
minikube start

# Déployer l'application
kubectl apply -f k8s/

# Vérifier les déploiements
kubectl get pods
kubectl get services
```

### Déploiement Cloud (Optionnel)

```bash
# Configurer kubectl pour votre cluster cloud
# Déployer avec Helm
helm install mlops-app ./helm/mlops-chart
```


## 🤝 Contributions

1. **Fork** le repository
2. **Créer** une branche feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** vos changements (`git commit -m 'Add some AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. **Ouvrir** une Pull Request

## 📝 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Équipe

- **Nom du Team** : TOUSSI Manoël Malaury
- **Contact** : [manoel@malaurytoussi.cm]


## 📚 Ressources Supplémentaires

### Choix Techniques

**Pourquoi TensorFlow ?**
- Excellent support pour la classification d'images
- Intégration native avec TensorFlow Serving
- Écosystème mature pour la production

**Pourquoi FastAPI ?**
- Performance élevée (basé sur Starlette)
- Documentation automatique (Swagger)
- Validation des données native

**Pourquoi Airflow ?**
- Orchestration robuste des pipelines
- Interface graphique intuitive
- Gestion des dépendances complexes

### Optimisations Réalisées

1. **Modèle** :
   - Transfer learning avec MobileNetV2
   - Augmentation des données
   - Early stopping et callbacks

2. **API** :
   - Mise en cache des modèles
   - Traitement asynchrone
   - Validation des entrées

3. **Infrastructure** :
   - Répartition de charge
   - Monitoring proactif
   - Scaling automatique

---

*Ce README sera maintenu à jour avec les évolutions du projet. N'hésitez pas à contribuer à son amélioration !*