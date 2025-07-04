import pytest
import time
import requests
import concurrent.futures
from unittest.mock import Mock

class TestPerformance:
    """Tests de performance"""
    
    def test_api_response_time(self):
        """Test du temps de réponse de l'API"""
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 2.0  # Moins de 2 secondes
            assert response.status_code == 200
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API non disponible pour tests de performance")
    
    def test_prediction_performance(self):
        """Test de performance des prédictions"""
        try:
            test_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg"
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/predict-url",
                params={"image_url": test_url},
                timeout=30
            )
            end_time = time.time()
            
            prediction_time = end_time - start_time
            assert prediction_time < 10.0  # Moins de 10 secondes
            assert response.status_code == 200
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API non disponible pour tests de performance")
    
    def test_concurrent_predictions(self):
        """Test de prédictions concurrentes"""
        try:
            def make_prediction():
                test_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000000.jpg"
                response = requests.post(
                    "http://localhost:8000/predict-url",
                    params={"image_url": test_url},
                    timeout=30
                )
                return response.status_code == 200
            
            # Lancer 5 prédictions en parallèle
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_prediction) for _ in range(5)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Toutes les prédictions doivent réussir
            assert all(results)
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API non disponible pour tests de concurrence")
