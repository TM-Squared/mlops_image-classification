import os
import sys
import mlflow
import tensorflow as tf
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# Configuration TensorFlow
tf.config.set_visible_devices([], 'GPU')

# Ajouter le chemin du modèle
sys.path.append('/opt/airflow/ml/models')
from simple_model import train_quick_model, train_model_from_minio, MinIOModelManager

def get_model_info_safe(minio_manager, model_name="plant_classifier"):
    """Obtenir les informations du modèle de manière sécurisée"""
    try:
        models_list = minio_manager.list_models(model_name)
        if models_list:
            return models_list[0]  # Le plus récent
        else:
            return {
                'key': 'model_not_found',
                'size': 0,
                'last_modified': datetime.now().isoformat(),
                'format': 'unknown'
            }
    except Exception as e:
        print(f"⚠️ Erreur récupération info modèle: {e}")
        return {
            'key': 'error_getting_info',
            'size': 0,
            'last_modified': datetime.now().isoformat(),
            'format': 'unknown'
        }

# Modifier les fonctions pour utiliser get_model_info_safe
def train_from_s3_keys(s3_keys, labels, num_epochs=3):
    """Entraîne le modèle avec des clés S3 spécifiques"""
    
    print(f"🎯 Entraînement avec {len(s3_keys)} images spécifiques depuis MinIO")
    
    # Configurer MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    
    # Entraîner le modèle
    model, accuracy = train_model_from_minio(s3_keys, labels, num_epochs=num_epochs)
    
    # Obtenir les informations du modèle sauvegardé
    minio_manager = MinIOModelManager()
    model_info = get_model_info_safe(minio_manager, "plant_classifier")
    
    return {
        'model_info': model_info,
        'accuracy': accuracy,
        'num_samples': len(s3_keys),
        'data_source': 'MinIO Custom Keys',
        'storage': 'MinIO'
    }

def train_from_database_minio(num_epochs=3):
    """Entraîne le modèle avec les données de la base"""
    
    # Récupérer les clés S3 depuis la base de données
    try:
        from airflow.providers.mysql.hooks.mysql import MySqlHook
        
        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        
        # Récupérer les données avec leurs clés S3
        query = """
        SELECT url_s3, label 
        FROM plants_data 
        WHERE url_s3 IS NOT NULL 
        AND image_exists = TRUE
        ORDER BY RAND()
        LIMIT 60
        """
        
        df = mysql_hook.get_pandas_df(query)
        
        if df.empty:
            print("❌ Aucune donnée trouvée dans la base, utilisation des données par défaut")
            return train_from_default_data(num_epochs)
        
        # Convertir les URLs S3 en clés S3
        s3_keys = []
        labels = []
        
        for _, row in df.iterrows():
            url_s3 = row['url_s3']
            label = row['label']
            
            # Convertir s3://raw-data/raw/dandelion/00000000.jpg en raw/dandelion/00000000.jpg
            if url_s3.startswith('s3://raw-data/'):
                s3_key = url_s3.replace('s3://raw-data/', '')
                s3_keys.append(s3_key)
                labels.append(label)
        
        print(f"📊 Données récupérées depuis la base:")
        print(f"  - Total: {len(s3_keys)} images")
        print(f"  - Pissenlits: {labels.count('dandelion')}")
        print(f"  - Herbe: {labels.count('grass')}")
        
        # Configurer MLflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        
        # Entraîner le modèle avec les données de MinIO
        model, accuracy = train_model_from_minio(s3_keys, labels, num_epochs=num_epochs)
        
        # Obtenir les informations du modèle sauvegardé
        minio_manager = MinIOModelManager()
        models_list = minio_manager.list_models("plant_classifier")
        
        latest_model = None
        if models_list:
            latest_model = models_list[0]  # Le plus récent
        
        return {
            'model_info': latest_model,
            'accuracy': accuracy,
            'num_samples': len(s3_keys),
            'data_source': 'MinIO via Database',
            'storage': 'MinIO'
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de l'accès à la base de données: {e}")
        print("🔄 Utilisation des données par défaut")
        return train_from_default_data(num_epochs)

def train_from_default_data(num_epochs=3):
    """Entraîne avec des données par défaut si la base n'est pas accessible"""
    
    # URLs par défaut (fallback)
    base_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
    
    dandelion_urls = [f"{base_url}/dandelion/{i:08d}.jpg" for i in range(30)]
    grass_urls = [f"{base_url}/grass/{i:08d}.jpg" for i in range(30)]
    
    all_urls = dandelion_urls + grass_urls
    all_labels = ['dandelion'] * 30 + ['grass'] * 30
    
    print(f"📊 Utilisation des données par défaut:")
    print(f"  - Total: {len(all_urls)} images")
    print(f"  - Pissenlits: {len(dandelion_urls)}")
    print(f"  - Herbe: {len(grass_urls)}")
    
    # Configurer MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    
    # Entraîner le modèle
    model, accuracy = train_quick_model(all_urls, all_labels, num_epochs=num_epochs)
    
    # Obtenir les informations du modèle sauvegardé
    minio_manager = MinIOModelManager()
    models_list = minio_manager.list_models("plant_classifier")
    
    latest_model = None
    if models_list:
        latest_model = models_list[0]  # Le plus récent
    
    return {
        'model_info': latest_model,
        'accuracy': accuracy,
        'num_samples': len(all_urls),
        'data_source': 'URLs Default Data',
        'storage': 'MinIO'
    }

def train_from_s3_keys(s3_keys, labels, num_epochs=3):
    """Entraîne le modèle avec des clés S3 spécifiques"""
    
    print(f"🎯 Entraînement avec {len(s3_keys)} images spécifiques depuis MinIO")
    
    # Configurer MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    
    # Entraîner le modèle
    model, accuracy = train_model_from_minio(s3_keys, labels, num_epochs=num_epochs)
    
    # Obtenir les informations du modèle sauvegardé
    minio_manager = MinIOModelManager()
    models_list = minio_manager.list_models("plant_classifier")
    
    latest_model = None
    if models_list:
        latest_model = models_list[0]  # Le plus récent
    
    return {
        'model_info': latest_model,
        'accuracy': accuracy,
        'num_samples': len(s3_keys),
        'data_source': 'MinIO Custom Keys',
        'storage': 'MinIO'
    }

def get_model_info(model_name="plant_classifier"):
    """Obtenir les informations sur les modèles disponibles"""
    minio_manager = MinIOModelManager()
    models_list = minio_manager.list_models(model_name)
    
    return {
        'available_models': models_list,
        'total_models': len(models_list),
        'storage': 'MinIO'
    }

# Fonctions de compatibilité
def train_from_database(num_epochs=3):
    """Fonction de compatibilité - redirige vers la version MinIO"""
    return train_from_database_minio(num_epochs)

def train_from_urls(image_urls, labels, num_epochs=3):
    """Fonction de compatibilité - utilise les URLs directement"""
    print("🔄 Entraînement avec URLs")
    
    # Configurer MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    
    # Entraîner le modèle
    model, accuracy = train_quick_model(image_urls, labels, num_epochs=num_epochs)
    
    # Obtenir les informations du modèle sauvegardé
    minio_manager = MinIOModelManager()
    models_list = minio_manager.list_models("plant_classifier")
    
    latest_model = None
    if models_list:
        latest_model = models_list[0]  # Le plus récent
    
    return {
        'model_info': latest_model,
        'accuracy': accuracy,
        'num_samples': len(image_urls),
        'data_source': 'URLs',
        'storage': 'MinIO'
    }

if __name__ == "__main__":
    result = train_from_database_minio()
    print(f"Résultat entraînement: {result}")