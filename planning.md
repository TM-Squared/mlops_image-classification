## Plan du Projet MLOps - Classification d'images

### Jour 1 — Initialisation & setup du projet
-  **Structuration du repo GitHub** : Choix entre monorepo ou multi-repo avec dossiers (`api/`, `webapp/`, `ml/`, `airflow/`, etc.).
-  **Environnement local avec Docker Compose** : MySQL, Minio, MLflow, Airflow.
-  **Script d'insertion SQL pour plants_data** : Préparation des données initiales.
-  **Scraping & ingestion des images (Airflow DAG)** : Récupération des images depuis URLs vers S3.
-  **Définir la structure de données** : raw/fmt/agg pour une logique data-centric.

### Jour 2 — Préprocessing & entraînement modèle
-  **Téléchargement des images depuis le bucket** : Via script ou Airflow.
-  **Pipeline de prétraitement** : Redimensionnement, normalisation, split train/test.
-  **Entraînement d’un modèle simple (CNN)** : Utilisation de PyTorch ou FastAI.
-  **Tracking des expériences avec MLflow** : Suivi des expériences.
-  **Sauvegarde du modèle dans Minio** : Stockage des modèles entraînés.

### Jour 3 — Packaging & CI/CD (partie entraînement)
-  **Création de scripts de training modulaires** : `train.py`, `utils.py`, etc.
-  **CI/CD pour tests et entraînement automatique** : Mise en place sur GitHub Actions.
-  **Airflow DAG pour réentraînement périodique** : Déclenchement par nouvelles données.
-  **Export des metrics vers MLflow** : Suivi des métriques d’entraînement.

### Jour 4 — API de prédiction
-  **Développement de l’API avec FastAPI** : API from scratch.
-  **Endpoint** : Upload image → réponse JSON avec prédiction.
-  **Intégration avec le modèle sauvegardé** : Chargement depuis S3/Minio.
-  **Tests unitaires de l’API** : Validation de l’API.
-  **Dockerisation de l’API** : Conteneurisation de l’API.

### Jour 5 — WebApp from scratch
-  **Création de la webapp interactive** : Utilisation de Streamlit ou développement frontend custom avec FastAPI.
-  **Upload d’image depuis l’interface** : Appel API → Affichage du résultat.
-  **Optionnel** : Affichage de l’image + prédiction + probabilité.
-  **Tests end-to-end simples** : Validation complète du workflow.

### Jour 6 — Déploiement Kubernetes local + Monitoring
-  **Déploiement Kubernetes local** : Utilisation de MiniKube ou Docker Desktop.
-  **Déploiement de l’API et webapp avec Helm** : Ou YAML + kubectl.
-  **Monitoring avec Grafana/Prometheus ou ELK** : Suivi de l’infrastructure.
-  **Optionnel** : Test de charge avec Locust.

### Jour 7 — Finalisation, doc & polish
-  **README complet** : Choix techniques, résultats, et captures d’écran.
-  **Documentation complète des modules** : Schéma d’architecture.
-  **Tests automatisés** : Unitaires + intégration.
-  **Push final GitHub + DockerHub** : Finalisation du dépôt.
-  **Préparation du mail avec livrables** : Format prêt pour l’envoi.
