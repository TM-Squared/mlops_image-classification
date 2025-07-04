import pytest
import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import io
import boto3
from moto import mock_s3
import mysql.connector
from unittest.mock import Mock, patch
import tensorflow as tf
from sqlalchemy import create_engine
import pandas as pd

# Configuration TensorFlow pour les tests
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@pytest.fixture(scope="session")
def test_config():
    """Configuration de test"""
    return {
        'mysql_host': os.getenv('MYSQL_HOST', 'mysql'),
        'mysql_user': os.getenv('MYSQL_USER', 'plants_user'),
        'mysql_password': os.getenv('MYSQL_PASSWORD', 'plants123'),
        'mysql_database': os.getenv('MYSQL_DATABASE', 'plants'),
        'api_url': os.getenv('API_URL', 'http://api:8000'),
        'minio_endpoint': os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
        'mlflow_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    }

@pytest.fixture
def temp_dir():
    """Créer un répertoire temporaire pour les tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_image():
    """Créer une image de test"""
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    return img_buffer.getvalue()

@pytest.fixture
def sample_images_data():
    """Données d'exemple pour les tests"""
    return {
        'image_urls': [
            'https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg',
            'https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000001.jpg',
            'https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000000.jpg',
            'https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000001.jpg'
        ],
        'labels': ['dandelion', 'dandelion', 'grass', 'grass'],
        's3_keys': [
            'raw/dandelion/00000000.jpg',
            'raw/dandelion/00000001.jpg',
            'raw/grass/00000000.jpg',
            'raw/grass/00000001.jpg'
        ]
    }

@pytest.fixture
def mock_s3_client():
    """Mock du client S3/MinIO"""
    with mock_s3():
        client = boto3.client(
            's3',
            region_name='us-east-1',
            aws_access_key_id='test',
            aws_secret_access_key='test'
        )
        
        # Créer les buckets de test
        client.create_bucket(Bucket='raw-data')
        client.create_bucket(Bucket='models')
        client.create_bucket(Bucket='mlflow')
        
        yield client

@pytest.fixture
def real_mysql_connection(test_config):
    """Connexion MySQL réelle pour les tests d'intégration"""
    try:
        connection = mysql.connector.connect(
            host=test_config['mysql_host'],
            user=test_config['mysql_user'],
            password=test_config['mysql_password'],
            database=test_config['mysql_database']
        )
        yield connection
        connection.close()
    except mysql.connector.Error:
        pytest.skip("MySQL non disponible pour les tests d'intégration")

@pytest.fixture
def mock_model():
    """Mock d'un modèle TensorFlow"""
    model = Mock()
    model.predict.return_value = np.array([[0.3, 0.7]])
    model.save = Mock()
    model.summary = Mock()
    return model