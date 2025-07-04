import pytest
import requests
import json
import time

class TestAPIInDocker:
    """Tests de l'API dans l'environnement Docker"""

    @pytest.fixture
    def api_base_url(self, test_config):
        return test_config['api_url']

    def test_api_root(self, api_base_url):
        """Test de l'endpoint racine"""
        try:
            response = requests.get(f"{api_base_url}/", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "message" in data
            assert "version" in data
            assert "framework" in data
            assert data["framework"] == "TensorFlow"
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Test API root échoué: {e}")

    def test_api_model_info(self, api_base_url):
        """Test des informations sur le modèle"""
        try:
            response = requests.get(f"{api_base_url}/model-info", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "model_type" in data
            assert "classes" in data
            assert "input_size" in data
            assert data["classes"] == ["grass", "dandelion"]
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Test model-info échoué: {e}")

    def test_api_predict_url(self, api_base_url):
        """Test de prédiction via URL"""
        try:
            test_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg"
            
            response = requests.post(
                f"{api_base_url}/predict-url",
                params={"image_url": test_url},
                timeout=30
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data
            
            # Vérifier les valeurs
            assert data["predicted_class"] in ["grass", "dandelion"]
            assert 0 <= data["confidence"] <= 1
            assert "grass" in data["probabilities"]
            assert "dandelion" in data["probabilities"]
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Test predict-url échoué: {e}")

    def test_api_models_list(self, api_base_url):
        """Test de la liste des modèles"""
        try:
            response = requests.get(f"{api_base_url}/models", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "available_models" in data
            assert "total_models" in data
            assert isinstance(data["available_models"], list)
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Test models list échoué: {e}")

    def test_api_reload_model(self, api_base_url):
        """Test de rechargement du modèle"""
        try:
            response = requests.post(f"{api_base_url}/reload-model", timeout=15)
            assert response.status_code == 200
            
            data = response.json()
            assert "message" in data
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Test reload-model échoué: {e}")
