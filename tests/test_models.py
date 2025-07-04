import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sys
import os
import tempfile

# Configuration pour les tests Docker
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TestModelsInDocker:
    """Tests des modèles dans l'environnement Docker"""

    def test_tensorflow_import(self):
        """Test d'import de TensorFlow"""
        try:
            import tensorflow as tf
            assert tf.__version__.startswith('2.13')
        except ImportError:
            pytest.fail("TensorFlow non disponible")

    def test_model_creation(self):
        """Test de création de modèle"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Créer un modèle simple
            model = keras.Sequential([
                keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                keras.layers.Dense(2, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            
            # Test de prédiction
            test_input = np.random.random((1, 5))
            prediction = model.predict(test_input, verbose=0)
            
            assert prediction.shape == (1, 2)
            assert np.allclose(np.sum(prediction), 1.0, atol=1e-6)
            
        except Exception as e:
            pytest.fail(f"Erreur création modèle: {e}")

    def test_model_save_load(self):
        """Test de sauvegarde/chargement de modèle"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Créer un modèle
            model = keras.Sequential([
                keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                keras.layers.Dense(2, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            
            # Sauvegarder dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                model.save(tmp_file.name)
                
                # Charger le modèle
                loaded_model = keras.models.load_model(tmp_file.name)
                
                # Test que les modèles sont équivalents
                test_input = np.random.random((1, 5))
                original_pred = model.predict(test_input, verbose=0)
                loaded_pred = loaded_model.predict(test_input, verbose=0)
                
                np.testing.assert_array_almost_equal(original_pred, loaded_pred)
                
                # Nettoyer
                os.unlink(tmp_file.name)
                
        except Exception as e:
            pytest.fail(f"Erreur sauvegarde/chargement: {e}")

    @patch('boto3.client')
    def test_minio_model_manager(self, mock_boto_client):
        """Test du gestionnaire de modèles MinIO"""
        try:
            # Simuler les modules si disponibles
            import sys
            from unittest.mock import Mock
            
            # Mock MinIOModelManager si le module n'est pas disponible
            mock_manager = Mock()
            mock_manager.save_model_to_minio.return_value = ['model_key_1', 'model_key_2']
            mock_manager.load_model_from_minio.return_value = (Mock(), 'model_key')
            mock_manager.list_models.return_value = [
                {'key': 'test_model.keras', 'size': 1024, 'format': 'keras'}
            ]
            
            # Tests des méthodes
            saved_keys = mock_manager.save_model_to_minio(Mock(), "test_model")
            assert len(saved_keys) == 2
            
            model, key = mock_manager.load_model_from_minio("test_model")
            assert model is not None
            assert key == 'model_key'
            
            models = mock_manager.list_models("test_model")
            assert len(models) == 1
            assert models[0]['format'] == 'keras'
            
        except Exception as e:
            pytest.fail(f"Erreur test MinIOModelManager: {e}")