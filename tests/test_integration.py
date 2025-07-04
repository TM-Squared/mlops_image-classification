import pytest
import requests
import time
import json
from sqlalchemy import create_engine, text
import mysql.connector

class TestDockerIntegration:
    """Tests d'intégration dans l'environnement Docker"""

    def test_mysql_connection(self, test_config):
        """Test de connexion à MySQL"""
        try:
            connection = mysql.connector.connect(
                host=test_config['mysql_host'],
                user=test_config['mysql_user'],
                password=test_config['mysql_password'],
                database=test_config['mysql_database']
            )
            
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            assert result[0] == 1
            
            cursor.close()
            connection.close()
            
        except mysql.connector.Error as e:
            pytest.fail(f"Connexion MySQL échouée: {e}")

    def test_minio_connection(self, test_config):
        """Test de connexion à MinIO"""
        try:
            import boto3
            
            s3_client = boto3.client(
                's3',
                endpoint_url=test_config['minio_endpoint'],
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin123',
                region_name='us-east-1'
            )
            
            # Lister les buckets
            response = s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            
            # Vérifier que les buckets existent ou peuvent être créés
            for bucket in ['raw-data', 'models', 'mlflow']:
                if bucket not in buckets:
                    s3_client.create_bucket(Bucket=bucket)
            
            # Test d'écriture/lecture
            s3_client.put_object(
                Bucket='raw-data',
                Key='test/test.txt',
                Body=b'test content'
            )
            
            response = s3_client.get_object(Bucket='raw-data', Key='test/test.txt')
            content = response['Body'].read()
            
            assert content == b'test content'
            
        except Exception as e:
            pytest.fail(f"Connexion MinIO échouée: {e}")

    def test_mlflow_connection(self, test_config):
        """Test de connexion à MLflow"""
        try:
            response = requests.get(f"{test_config['mlflow_uri']}/health", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Connexion MLflow échouée: {e}")

    def test_api_health(self, test_config):
        """Test de santé de l'API"""
        try:
            response = requests.get(f"{test_config['api_url']}/health", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"API non disponible: {e}")

    def test_api_prediction(self, test_config):
        """Test de prédiction via API"""
        try:
            test_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg"
            
            response = requests.post(
                f"{test_config['api_url']}/predict-url",
                params={"image_url": test_url},
                timeout=30
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert data["predicted_class"] in ["grass", "dandelion"]
            assert 0 <= data["confidence"] <= 1
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Test prédiction API échoué: {e}")

    def test_database_data(self, test_config):
        """Test des données dans la base"""
        try:
            connection = mysql.connector.connect(
                host=test_config['mysql_host'],
                user=test_config['mysql_user'],
                password=test_config['mysql_password'],
                database=test_config['mysql_database']
            )
            
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM plants_data")
            count = cursor.fetchone()[0]
            
            assert count > 0, "La table plants_data devrait contenir des données"
            
            cursor.execute("SELECT DISTINCT label FROM plants_data")
            labels = [row[0] for row in cursor.fetchall()]
            
            assert 'dandelion' in labels or 'grass' in labels
            
            cursor.close()
            connection.close()
            
        except mysql.connector.Error as e:
            pytest.fail(f"Test données base échoué: {e}")