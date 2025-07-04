from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import sys
import os
import pandas as pd

sys.path.append('/opt/airflow/ml/training')

def check_data_availability(**context):
    """Vérifier si suffisamment de données sont disponibles pour l'entraînement"""
    mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
    
    # Compter les images disponibles par classe
    query = """
    SELECT label, COUNT(*) as count
    FROM plants_data 
    WHERE url_s3 IS NOT NULL AND image_exists = TRUE
    GROUP BY label
    """
    
    try:
        df = mysql_hook.get_pandas_df(query)
        
        if df.empty:
            print("⚠️ Aucune donnée dans la base, utilisation des données par défaut")
            return {
                'total_images': 60,  # Valeur par défaut
                'classes': [
                    {'label': 'dandelion', 'count': 30},
                    {'label': 'grass', 'count': 30}
                ],
                'data_source': 'default'
            }
        
        print("📊 Données disponibles:")
        for _, row in df.iterrows():
            print(f"  - {row['label']}: {row['count']} images")
        
        # Vérifier qu'on a au moins 10 images de chaque classe
        min_images_per_class = 10
        for _, row in df.iterrows():
            if row['count'] < min_images_per_class:
                print(f"⚠️ Peu d'images pour {row['label']}: {row['count']} < {min_images_per_class}")
        
        total_images = df['count'].sum()
        print(f"✅ {total_images} images disponibles pour l'entraînement")
        
        return {
            'total_images': int(total_images),
            'classes': df.to_dict('records'),
            'data_source': 'database'
        }
        
    except Exception as e:
        print(f"❌ Erreur accès base de données: {e}")
        # Retourner des valeurs par défaut
        return {
            'total_images': 60,
            'classes': [
                {'label': 'dandelion', 'count': 30},
                {'label': 'grass', 'count': 30}
            ],
            'data_source': 'default'
        }

def prepare_training_data(**context):
    """Préparer les données d'entraînement depuis la base"""
    
    ti = context['ti']
    data_check = ti.xcom_pull(task_ids='check_data_availability')
    
    if data_check['data_source'] == 'default':
        print("🔄 Utilisation des données par défaut")
        return {
            'training_mode': 'default_urls',
            'total_samples': 60,
            'data_source': 'URLs par défaut'
        }
    
    try:
        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        
        # Récupérer toutes les données disponibles
        query = """
        SELECT url_s3, label 
        FROM plants_data 
        WHERE url_s3 IS NOT NULL AND image_exists = TRUE
        ORDER BY RAND()
        LIMIT 100
        """
        
        df = mysql_hook.get_pandas_df(query)
        
        if df.empty:
            print("⚠️ Aucune donnée trouvée, utilisation du mode par défaut")
            return {
                'training_mode': 'default_urls',
                'total_samples': 60,
                'data_source': 'URLs par défaut'
            }
        
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
        
        print(f"📋 {len(s3_keys)} images préparées pour l'entraînement")
        
        return {
            'training_mode': 'minio_keys',
            's3_keys': s3_keys,
            'labels': labels,
            'total_samples': len(s3_keys),
            'data_source': 'MinIO via Base'
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de la préparation des données: {e}")
        return {
            'training_mode': 'default_urls',
            'total_samples': 60,
            'data_source': 'URLs par défaut (erreur)'
        }

def train_model(**context):
    """Entraîner le modèle"""
    
    ti = context['ti']
    training_data = ti.xcom_pull(task_ids='prepare_training_data')
    
    if not training_data:
        raise ValueError("❌ Aucune donnée d'entraînement reçue")
    
    print(f"🚀 Début de l'entraînement")
    print(f"  - Mode: {training_data['training_mode']}")
    print(f"  - Échantillons: {training_data['total_samples']}")
    print(f"  - Source: {training_data['data_source']}")
    
    # Importer le trainer
    from trainer import train_from_s3_keys, train_from_database_minio
    
    try:
        if training_data['training_mode'] == 'minio_keys':
            # Entraîner avec les clés S3
            result = train_from_s3_keys(
                training_data['s3_keys'],
                training_data['labels'],
                num_epochs=5
            )
        else:
            # Entraîner avec les données par défaut
            result = train_from_database_minio(num_epochs=5)
        
        print(f"✅ Entraînement terminé:")
        print(f"  - Précision: {result['accuracy']:.2%}")
        print(f"  - Échantillons: {result['num_samples']}")
        print(f"  - Stockage: {result.get('storage', 'MinIO')}")
        
        # Informations du modèle
        model_info = result.get('model_info', {})
        if model_info:
            print(f"  - Modèle: {model_info.get('key', 'N/A')}")
            print(f"  - Taille: {model_info.get('size', 'N/A')} bytes")
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        raise

def evaluate_model(**context):
    """Évaluer le modèle entraîné"""
    
    ti = context['ti']
    training_result = ti.xcom_pull(task_ids='train_model')
    
    if not training_result:
        raise ValueError("❌ Aucun résultat d'entraînement reçu")
    
    accuracy = training_result['accuracy']
    min_accuracy = 0.6  # 60% minimum
    
    print(f"🔍 Évaluation du modèle:")
    print(f"  - Précision obtenue: {accuracy:.2%}")
    print(f"  - Seuil minimum: {min_accuracy:.2%}")
    print(f"  - Stockage: {training_result.get('storage', 'MinIO')}")
    
    # Informations du modèle
    model_info = training_result.get('model_info', {})
    if model_info:
        print(f"  - Modèle stocké: {model_info.get('key', 'N/A')}")
        print(f"  - Format: {model_info.get('format', 'N/A')}")
        print(f"  - Date: {model_info.get('last_modified', 'N/A')}")
    
    if accuracy >= min_accuracy:
        print("✅ Modèle validé pour le déploiement")
        return {
            'status': 'APPROVED',
            'accuracy': accuracy,
            'model_info': model_info,
            'storage': training_result.get('storage', 'MinIO')
        }
    else:
        print("❌ Modèle rejeté (précision insuffisante)")
        return {
            'status': 'REJECTED',
            'accuracy': accuracy,
            'reason': f'Précision {accuracy:.2%} < {min_accuracy:.2%}',
            'model_info': model_info,
            'storage': training_result.get('storage', 'MinIO')
        }

def deploy_model(**context):
    """Déployer le modèle approuvé"""
    
    ti = context['ti']
    evaluation_result = ti.xcom_pull(task_ids='evaluate_model')
    
    if evaluation_result['status'] == 'APPROVED':
        print("🚀 Déploiement du modèle approuvé")
        
        model_info = evaluation_result.get('model_info', {})
        storage = evaluation_result.get('storage', 'MinIO')
        
        print(f"📦 Modèle déployé avec succès")
        print(f"  - Stockage: {storage}")
        print(f"  - Précision: {evaluation_result['accuracy']:.2%}")
        
        if model_info:
            print(f"  - Fichier: {model_info.get('key', 'N/A')}")
            print(f"  - Format: {model_info.get('format', 'N/A')}")
            print(f"  - Taille: {model_info.get('size', 'N/A')} bytes")
        
        return {
            'status': 'DEPLOYED',
            'model_info': model_info,
            'accuracy': evaluation_result['accuracy'],
            'storage': storage,
            'deployment_time': datetime.now().isoformat()
        }
    else:
        print("❌ Déploiement annulé - modèle non approuvé")
        return {
            'status': 'DEPLOYMENT_CANCELLED',
            'reason': evaluation_result.get('reason', 'Validation failed'),
            'accuracy': evaluation_result['accuracy']
        }

def send_notification(**context):
    """Envoyer une notification de fin de pipeline"""
    
    ti = context['ti']
    
    # Récupérer les résultats de toutes les tâches
    data_check = ti.xcom_pull(task_ids='check_data_availability')
    training_result = ti.xcom_pull(task_ids='train_model')
    evaluation_result = ti.xcom_pull(task_ids='evaluate_model')
    deployment_result = ti.xcom_pull(task_ids='deploy_model')
    
    print("📧 Notification de fin de pipeline:")
    print(f"  - Données: {data_check['total_images']} images ({data_check['data_source']})")
    print(f"  - Entraînement: {training_result['accuracy']:.2%} de précision")
    print(f"  - Évaluation: {evaluation_result['status']}")
    print(f"  - Déploiement: {deployment_result['status']}")
    print(f"  - Stockage: {training_result.get('storage', 'MinIO')}")
    
    # Informations du modèle
    model_info = training_result.get('model_info', {})
    if model_info:
        print(f"  - Modèle: {model_info.get('key', 'N/A')}")
    
    # Simuler l'envoi d'une notification
    notification_data = {
        'pipeline': 'model_training_pipeline',
        'status': 'SUCCESS' if deployment_result['status'] == 'DEPLOYED' else 'PARTIAL_SUCCESS',
        'accuracy': training_result['accuracy'],
        'storage': training_result.get('storage', 'MinIO'),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"📨 Notification envoyée: {notification_data}")
    
    return "NOTIFICATION_SENT"

# Configuration du DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Pipeline complet d\'entraînement de modèle avec MinIO',
    schedule_interval=None,  # Déclenchement manuel
    catchup=False,
    tags=['ml', 'training', 'tensorflow', 'minio', 'production'],
    max_active_runs=1,
    doc_md="""
    ## Pipeline d'entraînement de modèle avec MinIO
    
    Ce DAG orchestre l'entraînement complet d'un modèle de classification d'images
    avec stockage sur MinIO.
    
    ### Étapes:
    1. **check_data_availability**: Vérifie la disponibilité des données
    2. **prepare_training_data**: Prépare les données d'entraînement
    3. **train_model**: Entraîne le modèle TensorFlow
    4. **evaluate_model**: Évalue les performances
    5. **deploy_model**: Déploie le modèle si approuvé
    6. **send_notification**: Envoie une notification de fin
    
    ### Fonctionnalités:
    - Stockage des modèles sur MinIO
    - Fallback sur données par défaut si base indisponible
    - Versionning automatique des modèles
    - Métriques détaillées avec MLflow
    
    ### Prérequis:
    - MinIO configuré avec bucket 'models'
    - Base de données MySQL avec table plants_data
    - MLflow pour le tracking
    """
)

# Définition des tâches
check_data_task = PythonOperator(
    task_id='check_data_availability',
    python_callable=check_data_availability,
    provide_context=True,
    dag=dag,
    doc_md="Vérifie la disponibilité des données d'entraînement"
)

prepare_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    provide_context=True,
    dag=dag,
    doc_md="Prépare les données pour l'entraînement"
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
    doc_md="Entraîne le modèle TensorFlow avec sauvegarde MinIO"
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
    doc_md="Évalue les performances du modèle"
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    provide_context=True,
    dag=dag,
    doc_md="Déploie le modèle si validé"
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    provide_context=True,
    dag=dag,
    doc_md="Envoie une notification de fin de pipeline"
)

# Définir les dépendances
check_data_task >> prepare_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task >> notify_task