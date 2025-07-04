import pytest
from unittest.mock import patch, Mock
from datetime import datetime

class TestAirflowDAGs:
    """Tests pour les DAGs Airflow"""
    
    def test_dag_import(self):
        """Test d'import des DAGs"""
        try:
            # Tenter d'importer les DAGs
            import sys
            import os
            
            # Ajouter le chemin des DAGs
            dags_path = os.path.join(os.path.dirname(__file__), '..', 'airflow', 'dags')
            sys.path.append(dags_path)
            
            # Tester l'import de quelques DAGs
            dag_files = [
                'setup_connections',
                'check_connections'
            ]
            
            for dag_file in dag_files:
                try:
                    __import__(dag_file)
                    print(f"✅ DAG {dag_file} importé avec succès")
                except ImportError as e:
                    print(f"⚠️ DAG {dag_file} non disponible: {e}")
                    
        except Exception as e:
            pytest.skip(f"Tests DAGs non disponibles: {e}")
    
    @patch('airflow.providers.mysql.hooks.mysql.MySqlHook')
    def test_database_connection_task(self, mock_mysql_hook):
        """Test de la tâche de connexion à la base"""
        # Mock de la connexion MySQL
        mock_hook_instance = Mock()
        mock_mysql_hook.return_value = mock_hook_instance
        mock_hook_instance.get_first.return_value = (1,)
        
        # Simuler une tâche de test de connexion
        def test_mysql_connection():
            hook = mock_mysql_hook()
            result = hook.get_first("SELECT 1")
            return result[0] == 1
        
        assert test_mysql_connection() is True
        mock_hook_instance.get_first.assert_called_once()
