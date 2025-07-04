import pytest
from unittest.mock import patch, Mock, MagicMock
import pandas as pd

class TestTraining:
    """Tests pour l'entraînement des modèles"""
    
    @patch('ml.training.trainer.MySqlHook')
    @patch('ml.training.trainer.train_model_from_minio')
    def test_train_from_database_minio_success(self, mock_train, mock_mysql_hook):
        """Test d'entraînement réussi depuis la base"""
        # Configuration du mock MySQL
        mock_hook_instance = Mock()
        mock_mysql_hook.return_value = mock_hook_instance
        
        # Mock des données de la base
        mock_df = pd.DataFrame({
            'url_s3': ['s3://raw-data/raw/dandelion/test.jpg', 's3://raw-data/raw/grass/test.jpg'],
            'label': ['dandelion', 'grass']
        })
        mock_hook_instance.get_pandas_df.return_value = mock_df
        
        # Mock de l'entraînement
        mock_model = Mock()
        mock_train.return_value = (mock_model, 0.85)
        
        try:
            from ml.training.trainer import train_from_database_minio
            
            result = train_from_database_minio(num_epochs=1)
            
            assert result['accuracy'] == 0.85
            assert result['num_samples'] == 2
            assert result['data_source'] == 'MinIO via Database'
            mock_train.assert_called_once()
            
        except ImportError:
            pytest.skip("Modules d'entraînement non disponibles")
    
    @patch('ml.training.trainer.train_quick_model')
    def test_train_from_urls(self, mock_train):
        """Test d'entraînement depuis URLs"""
        # Mock de l'entraînement
        mock_model = Mock()
        mock_train.return_value = (mock_model, 0.80)
        
        try:
            from ml.training.trainer import train_from_urls
            
            urls = ['http://example.com/1.jpg', 'http://example.com/2.jpg']
            labels = ['dandelion', 'grass']
            
            result = train_from_urls(urls, labels, num_epochs=1)
            
            assert result['accuracy'] == 0.80
            assert result['num_samples'] == 2
            mock_train.assert_called_once()
            
        except ImportError:
            pytest.skip("Modules d'entraînement non disponibles")
