import os
import sys
from airflow.models import Connection
from airflow.utils.db import create_session
from airflow import settings
import json

def create_mysql_connection():
    """Créer la connexion MySQL"""
    connection_id = 'mysql_default'
    
    # Supprimer l'ancienne connexion si elle existe
    with create_session() as session:
        existing_conn = session.query(Connection).filter(Connection.conn_id == connection_id).first()
        if existing_conn:
            session.delete(existing_conn)
            session.commit()
            print(f"✅ Ancienne connexion {connection_id} supprimée")
    
    # Créer la nouvelle connexion
    new_conn = Connection(
        conn_id=connection_id,
        conn_type='mysql',
        host=os.getenv('MYSQL_HOST', 'mysql'),
        login=os.getenv('MYSQL_USER', 'plants_user'),
        password=os.getenv('MYSQL_PASSWORD', 'plants123'),
        schema=os.getenv('MYSQL_DATABASE', 'plants'),
        port=3306
    )
    
    with create_session() as session:
        session.add(new_conn)
        session.commit()
        print(f"✅ Connexion MySQL '{connection_id}' créée avec succès")

def create_s3_connection():
    """Créer la connexion S3/MinIO"""
    connection_id = 's3_connec'
    
    # Supprimer l'ancienne connexion si elle existe
    with create_session() as session:
        existing_conn = session.query(Connection).filter(Connection.conn_id == connection_id).first()
        if existing_conn:
            session.delete(existing_conn)
            session.commit()
            print(f"✅ Ancienne connexion {connection_id} supprimée")
    
    # Configuration pour MinIO
    minio_access_key = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
    minio_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123')
    minio_endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
    
    # Extra configuration pour S3/MinIO
    extra_config = {
        "endpoint_url": minio_endpoint,
        "aws_access_key_id": minio_access_key,
        "aws_secret_access_key": minio_secret_key,
        "region_name": "us-east-1"
    }
    
    # Créer la nouvelle connexion
    new_conn = Connection(
        conn_id=connection_id,
        conn_type='aws',
        login=minio_access_key,
        password=minio_secret_key,
        extra=json.dumps(extra_config)
    )
    
    with create_session() as session:
        session.add(new_conn)
        session.commit()
        print(f"✅ Connexion S3/MinIO '{connection_id}' créée avec succès")
        print(f"   - Endpoint: {minio_endpoint}")
        print(f"   - Access Key: {minio_access_key}")

def create_mlflow_connection():
    """Créer la connexion MLflow (optionnel)"""
    connection_id = 'mlflow_default'
    
    # Supprimer l'ancienne connexion si elle existe
    with create_session() as session:
        existing_conn = session.query(Connection).filter(Connection.conn_id == connection_id).first()
        if existing_conn:
            session.delete(existing_conn)
            session.commit()
            print(f"✅ Ancienne connexion {connection_id} supprimée")
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    
    # Créer la nouvelle connexion
    new_conn = Connection(
        conn_id=connection_id,
        conn_type='http',
        host=mlflow_uri
    )
    
    with create_session() as session:
        session.add(new_conn)
        session.commit()
        print(f"✅ Connexion MLflow '{connection_id}' créée avec succès")

def main():
    """Fonction principale"""
    print("🔧 Création des connexions Airflow...")
    
    try:
        # Créer les connexions
        create_mysql_connection()
        create_s3_connection()
        create_mlflow_connection()
        
        print("\n🎉 Toutes les connexions ont été créées avec succès!")
        
        # Afficher un résumé
        print("\n📋 Résumé des connexions créées:")
        print("  - mysql_default: Connexion MySQL pour la base de données")
        print("  - s3_connec: Connexion S3/MinIO pour le stockage")
        print("  - mlflow_default: Connexion MLflow pour le tracking")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des connexions: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()