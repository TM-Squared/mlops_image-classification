from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow/ml/training')

def check_new_data(**context):
    """Vérifier s'il y a de nouvelles données"""
    mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
    
    # Compter les nouvelles données depuis la dernière exécution
    query = """
    SELECT COUNT(*) as new_count
    FROM plants_data 
    WHERE url_s3 IS NOT NULL 
    AND image_exists = TRUE
    AND created_at > DATE_SUB(NOW(), INTERVAL 1 DAY)
    """
    
    try:
        result = mysql_hook.get_first(query)
        new_count = result[0] if result else 0
    except:
        # Si la colonne created_at n'existe pas, compter toutes les données
        query = """
        SELECT COUNT(*) as total_count
        FROM plants_data 
        WHERE url_s3 IS NOT NULL 
        AND image_exists = TRUE
        """
        result = mysql_hook.get_first(query)
        new_count = result[0] if result else 0
    
    print(f"📊 Nouvelles données détectées: {new_count}")
    
    min_new_data = 10  # Minimum pour déclencher un réentraînement
    
    if new_count >= min_new_data:
        print(f"✅ Assez de nouvelles données ({new_count} >= {min_new_data})")
        return True
    else:
        print(f"❌ Pas assez de nouvelles données ({new_count} < {min_new_data})")
        return False

def retrain_with_new_data(**context):
    """Réentraîner le modèle avec toutes les données disponibles"""
    print("🔄 Réentraînement avec nouvelles données")
    
    # Utiliser le même trainer que pour l'entraînement normal
    from trainer import train_from_database
    
    # Mais avec plus d'époques pour l'entraînement continu
    result = train_from_database(num_epochs=10)
    
    print(f"✅ Réentraînement terminé:")
    print(f"  - Précision: {result['accuracy']:.2%}")
    print(f"  - Échantillons: {result['num_samples']}")
    
    return result

def compare_model_performance(**context):
    """Comparer les performances du nouveau modèle"""
    ti = context['ti']
    retraining_result = ti.xcom_pull(task_ids='retrain_with_new_data')
    
    new_accuracy = retraining_result['accuracy']
    
    # Ici, on pourrait comparer avec l'ancien modèle
    # Pour simplifier, on considère que le nouveau modèle est meilleur si > 70%
    
    threshold = 0.7
    
    if new_accuracy > threshold:
        print(f"✅ Nouveau modèle accepté ({new_accuracy:.2%} > {threshold:.2%})")
        return {
            'decision': 'ACCEPT_NEW_MODEL',
            'new_accuracy': new_accuracy,
            'reason': f'Performance supérieure au seuil ({new_accuracy:.2%})'
        }
    else:
        print(f"❌ Nouveau modèle rejeté ({new_accuracy:.2%} <= {threshold:.2%})")
        return {
            'decision': 'REJECT_NEW_MODEL',
            'new_accuracy': new_accuracy,
            'reason': f'Performance insuffisante ({new_accuracy:.2%})'
        }

def deploy_new_model(**context):
    """Déployer le nouveau modèle si approuvé"""
    ti = context['ti']
    comparison_result = ti.xcom_pull(task_ids='compare_model_performance')
    
    if comparison_result['decision'] == 'ACCEPT_NEW_MODEL':
        print("🚀 Déploiement du nouveau modèle")
        
        # Le modèle est déjà dans /shared/models/ grâce au réentraînement
        # On pourrait ici faire une sauvegarde de l'ancien modèle
        
        print("📦 Nouveau modèle déployé avec succès")
        return {
            'status': 'DEPLOYED',
            'accuracy': comparison_result['new_accuracy']
        }
    else:
        print("❌ Nouveau modèle non déployé")
        return {
            'status': 'NOT_DEPLOYED',
            'reason': comparison_result['reason']
        }

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
    'continuous_training_pipeline',
    default_args=default_args,
    description='Pipeline d\'entraînement continu avec nouvelles données',
    schedule_interval=timedelta(days=1),  # Quotidien
    catchup=False,
    tags=['ml', 'continuous', 'training', 'automation'],
    max_active_runs=1
)

# Définition des tâches
check_new_data_task = PythonOperator(
    task_id='check_new_data',
    python_callable=check_new_data,
    provide_context=True,
    dag=dag
)

retrain_task = PythonOperator(
    task_id='retrain_with_new_data',
    python_callable=retrain_with_new_data,
    provide_context=True,
    dag=dag
)

compare_performance_task = PythonOperator(
    task_id='compare_model_performance',
    python_callable=compare_model_performance,
    provide_context=True,
    dag=dag
)

deploy_new_model_task = PythonOperator(
    task_id='deploy_new_model',
    python_callable=deploy_new_model,
    provide_context=True,
    dag=dag
)

# Définir les dépendances avec branchement conditionnel
check_new_data_task >> retrain_task >> compare_performance_task >> deploy_new_model_task