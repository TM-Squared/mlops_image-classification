from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import sys

def create_connections():
    """Créer les connexions Airflow"""
    print("🔧 Création des connexions Airflow...")
    
    # Exécuter le script de création des connexions
    sys.path.append('/opt/airflow/dags/scripts')
    
    try:
        # Importer et exécuter le script
        from create_connections import main as create_connections_main
        create_connections_main()
        print("✅ Connexions créées avec succès")
        return "SUCCESS"
    except Exception as e:
        print(f"❌ Erreur lors de la création des connexions: {e}")
        raise

def create_minio_buckets():
    """Créer les buckets MinIO"""
    print("🪣 Création des buckets MinIO...")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Configuration MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
            region_name='us-east-1'
        )
        
        # Buckets à créer
        buckets = ['raw-data', 'models', 'mlflow']
        
        for bucket in buckets:
            try:
                s3_client.create_bucket(Bucket=bucket)
                print(f'✅ Bucket {bucket} créé')
            except ClientError as e:
                if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                    print(f'✅ Bucket {bucket} existe déjà')
                else:
                    print(f'⚠️ Erreur création bucket {bucket}: {e}')
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des buckets: {e}")
        raise

def test_connections():
    """Tester les connexions créées"""
    print("🧪 Test des connexions...")
    
    try:
        from airflow.providers.mysql.hooks.mysql import MySqlHook
        from airflow.providers.amazon.aws.hooks.s3 import S3Hook
        
        # Test MySQL
        try:
            mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
            result = mysql_hook.get_first("SELECT 1 as test")
            print(f"✅ MySQL: {result}")
        except Exception as e:
            print(f"❌ MySQL: {e}")
        
        # Test S3
        try:
            s3_hook = S3Hook(aws_conn_id='s3_connec')
            buckets = s3_hook.list_buckets()
            print(f"✅ S3/MinIO: {len(buckets)} buckets trouvés")
        except Exception as e:
            print(f"❌ S3/MinIO: {e}")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"❌ Erreur lors du test des connexions: {e}")
        raise

# Configuration du DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'setup_connections',
    default_args=default_args,
    description='Configuration automatique des connexions Airflow',
    schedule_interval=None, 
    catchup=False,
    tags=['admin', 'setup', 'connections'],
    max_active_runs=1,
    doc_md="""
    ## Configuration des connexions Airflow
    
    Ce DAG configure automatiquement toutes les connexions nécessaires :
    - MySQL pour la base de données
    - S3/MinIO pour le stockage
    - MLflow pour le tracking
    
    **Utilisation :**
    1. Déclencher manuellement ce DAG après le premier démarrage
    2. Vérifier dans Admin > Connections que les connexions sont créées
    3. Les DAGs suivants pourront utiliser ces connexions
    """
)

# Définition des tâches
create_connections_task = PythonOperator(
    task_id='create_connections',
    python_callable=create_connections,
    dag=dag
)

create_buckets_task = PythonOperator(
    task_id='create_minio_buckets',
    python_callable=create_minio_buckets,
    dag=dag
)

test_connections_task = PythonOperator(
    task_id='test_connections',
    python_callable=test_connections,
    dag=dag
)

# Notification finale
notify_task = BashOperator(
    task_id='notify_completion',
    bash_command='echo "🎉 Configuration terminée ! Connexions prêtes à être utilisées."',
    dag=dag
)

# Définir les dépendances
create_connections_task >> create_buckets_task >> test_connections_task >> notify_task