from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import sys
import os
import requests
import json

sys.path.append('/opt/airflow/ml/models')
sys.path.append('/opt/airflow/ml/training')

def load_test_data(**context):
    """Charger des données de test"""
    try:
        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        
        # Prendre 20 images de test
        query = """
        SELECT url_s3, label 
        FROM plants_data 
        WHERE url_s3 IS NOT NULL AND image_exists = TRUE
        ORDER BY RAND()
        LIMIT 20
        """
        
        df = mysql_hook.get_pandas_df(query)
        
        if df.empty:
            print("⚠️ Aucune donnée de test dans la base, utilisation d'URLs par défaut")
            
            # URLs par défaut
            base_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
            test_data = [
                {"url": f"{base_url}/dandelion/00000010.jpg", "label": "dandelion"},
                {"url": f"{base_url}/dandelion/00000011.jpg", "label": "dandelion"},
                {"url": f"{base_url}/dandelion/00000012.jpg", "label": "dandelion"},
                {"url": f"{base_url}/grass/00000010.jpg", "label": "grass"},
                {"url": f"{base_url}/grass/00000011.jpg", "label": "grass"},
                {"url": f"{base_url}/grass/00000012.jpg", "label": "grass"},
            ]
            
            return {
                'test_data': test_data,
                'data_source': 'default_urls'
            }
        
        # Convertir les URLs S3 en URLs HTTP pour les tests
        test_data = []
        for _, row in df.iterrows():
            url_s3 = row['url_s3']
            label = row['label']
            
            # Convertir s3://raw-data/raw/dandelion/00000000.jpg en URL HTTP
            if url_s3.startswith('s3://raw-data/'):
                # Pour les tests, utiliser les URLs GitHub à la place
                filename = url_s3.split('/')[-1]
                category = url_s3.split('/')[-2]
                base_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
                http_url = f"{base_url}/{category}/{filename}"
                
                test_data.append({
                    "url": http_url,
                    "label": label,
                    "s3_key": url_s3.replace('s3://raw-data/', '')
                })
        
        print(f"📋 {len(test_data)} images de test chargées depuis la base")
        
        return {
            'test_data': test_data,
            'data_source': 'database'
        }
        
    except Exception as e:
        print(f"❌ Erreur chargement données test: {e}")
        
        # Fallback vers des URLs par défaut
        base_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
        test_data = [
            {"url": f"{base_url}/dandelion/00000010.jpg", "label": "dandelion"},
            {"url": f"{base_url}/grass/00000010.jpg", "label": "grass"},
        ]
        
        return {
            'test_data': test_data,
            'data_source': 'fallback_urls'
        }

def evaluate_current_model(**context):
    """Évaluer le modèle actuel via l'API"""
    ti = context['ti']
    test_result = ti.xcom_pull(task_ids='load_test_data')
    
    if not test_result:
        raise ValueError("❌ Aucune donnée de test reçue")
    
    test_data = test_result['test_data']
    data_source = test_result['data_source']
    
    print(f"🔍 Évaluation du modèle avec {len(test_data)} images")
    print(f"📊 Source des données: {data_source}")
    
    # Tester la disponibilité de l'API
    try:
        response = requests.get("http://api:8000/health", timeout=10)
        if response.status_code != 200:
            raise Exception(f"API non disponible: {response.status_code}")
        
        api_info = response.json()
        print(f"✅ API accessible: {api_info.get('framework', 'N/A')} - {api_info.get('tf_version', 'N/A')}")
        
    except Exception as e:
        print(f"❌ API non accessible: {e}")
        print("🔄 Tentative d'évaluation locale...")
        return evaluate_model_locally(test_data)
    
    # Évaluer via l'API
    return evaluate_model_via_api(test_data)

def evaluate_model_via_api(test_data):
    """Évaluer le modèle via l'API"""
    
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    failed_requests = 0
    
    print("🌐 Évaluation via API")
    
    for i, test_item in enumerate(test_data):
        try:
            # Faire une prédiction via l'API
            response = requests.post(
                "http://api:8000/predict-url",
                json={"image_url": test_item["url"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted_label = result['predicted_class']
                true_label = test_item['label']
                confidence = result['confidence']
                
                is_correct = predicted_label == true_label
                if is_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                predictions.append({
                    'test_id': i + 1,
                    'url': test_item['url'],
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
                print(f"Test {i+1}: {true_label} -> {predicted_label} ({'✅' if is_correct else '❌'}) - {confidence:.2%}")
                
            else:
                print(f"❌ Erreur API pour test {i+1}: {response.status_code}")
                failed_requests += 1
                
        except Exception as e:
            print(f"❌ Erreur test {i+1}: {e}")
            failed_requests += 1
    
    # Calculer les métriques
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    success_rate = total_predictions / len(test_data) if len(test_data) > 0 else 0
    
    print(f"\n📊 Résultats de l'évaluation API:")
    print(f"  - Tests réussis: {total_predictions}/{len(test_data)} ({success_rate:.2%})")
    print(f"  - Prédictions correctes: {correct_predictions}/{total_predictions}")
    print(f"  - Précision: {accuracy:.2%}")
    print(f"  - Requêtes échouées: {failed_requests}")
    
    return {
        'evaluation_method': 'api',
        'accuracy': accuracy,
        'total_tests': len(test_data),
        'successful_tests': total_predictions,
        'correct_predictions': correct_predictions,
        'failed_requests': failed_requests,
        'success_rate': success_rate,
        'predictions': predictions
    }

def evaluate_model_locally(test_data):
    """Évaluer le modèle localement (fallback)"""
    
    print("🏠 Évaluation locale du modèle")
    
    try:
        # Importer les fonctions nécessaires
        from simple_model import load_model_for_prediction, predict_image_from_url
        
        # Charger le modèle
        model = load_model_for_prediction("plant_classifier", "latest")
        
        if model is None:
            print("❌ Impossible de charger le modèle pour l'évaluation locale")
            return {
                'evaluation_method': 'local',
                'accuracy': 0.0,
                'total_tests': len(test_data),
                'successful_tests': 0,
                'correct_predictions': 0,
                'error': 'Modèle non disponible'
            }
        
        print("✅ Modèle chargé pour l'évaluation locale")
        
        correct_predictions = 0
        total_predictions = 0
        predictions = []
        
        for i, test_item in enumerate(test_data):
            try:
                # Faire une prédiction locale
                result = predict_image_from_url(model, test_item['url'])
                
                if result:
                    predicted_label = result['predicted_class']
                    true_label = test_item['label']
                    confidence = result['confidence']
                    
                    is_correct = predicted_label == true_label
                    if is_correct:
                        correct_predictions += 1
                    
                    total_predictions += 1
                    
                    predictions.append({
                        'test_id': i + 1,
                        'url': test_item['url'],
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                    
                    print(f"Test {i+1}: {true_label} -> {predicted_label} ({'✅' if is_correct else '❌'}) - {confidence:.2%}")
                    
                else:
                    print(f"❌ Erreur prédiction locale pour test {i+1}")
                    
            except Exception as e:
                print(f"❌ Erreur test local {i+1}: {e}")
        
        # Calculer les métriques
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        success_rate = total_predictions / len(test_data) if len(test_data) > 0 else 0
        
        print(f"\n📊 Résultats de l'évaluation locale:")
        print(f"  - Tests réussis: {total_predictions}/{len(test_data)} ({success_rate:.2%})")
        print(f"  - Prédictions correctes: {correct_predictions}/{total_predictions}")
        print(f"  - Précision: {accuracy:.2%}")
        
        return {
            'evaluation_method': 'local',
            'accuracy': accuracy,
            'total_tests': len(test_data),
            'successful_tests': total_predictions,
            'correct_predictions': correct_predictions,
            'success_rate': success_rate,
            'predictions': predictions
        }
        
    except Exception as e:
        print(f"❌ Erreur évaluation locale: {e}")
        return {
            'evaluation_method': 'local',
            'accuracy': 0.0,
            'total_tests': len(test_data),
            'successful_tests': 0,
            'correct_predictions': 0,
            'error': str(e)
        }

def check_model_performance(**context):
    """Vérifier si les performances sont acceptables"""
    ti = context['ti']
    eval_result = ti.xcom_pull(task_ids='evaluate_current_model')
    
    if not eval_result:
        raise ValueError("❌ Aucun résultat d'évaluation reçu")
    
    accuracy = eval_result['accuracy']
    method = eval_result['evaluation_method']
    success_rate = eval_result.get('success_rate', 1.0)
    
    min_accuracy = 0.70  # 70% minimum pour la production
    min_success_rate = 0.80  # 80% des tests doivent réussir
    
    print(f"🔍 Analyse des performances:")
    print(f"  - Méthode d'évaluation: {method}")
    print(f"  - Précision: {accuracy:.2%}")
    print(f"  - Taux de réussite: {success_rate:.2%}")
    print(f"  - Seuil précision: {min_accuracy:.2%}")
    print(f"  - Seuil réussite: {min_success_rate:.2%}")
    
    # Déterminer le statut
    if accuracy >= min_accuracy and success_rate >= min_success_rate:
        status = "PERFORMANCE_OK"
        message = "✅ Performances acceptables"
        needs_retraining = False
    elif accuracy < min_accuracy:
        status = "PERFORMANCE_DEGRADED"
        message = "⚠️ Précision insuffisante - Réentraînement recommandé"
        needs_retraining = True
    else:
        status = "SYSTEM_ISSUES"
        message = "⚠️ Problèmes techniques détectés"
        needs_retraining = False
    
    print(f"  - Status: {message}")
    
    # Recommandations
    recommendations = []
    if accuracy < min_accuracy:
        recommendations.append("Réentraîner le modèle avec plus de données")
    if success_rate < min_success_rate:
        recommendations.append("Vérifier la connectivité et la stabilité de l'API")
    if not recommendations:
        recommendations.append("Continuer la surveillance")
    
    return {
        'status': status,
        'accuracy': accuracy,
        'success_rate': success_rate,
        'needs_retraining': needs_retraining,
        'evaluation_method': method,
        'recommendations': recommendations,
        'details': eval_result
    }

def generate_report(**context):
    """Générer un rapport d'évaluation"""
    ti = context['ti']
    
    # Récupérer les résultats
    test_data_result = ti.xcom_pull(task_ids='load_test_data')
    eval_result = ti.xcom_pull(task_ids='evaluate_current_model')
    performance_result = ti.xcom_pull(task_ids='check_model_performance')
    
    print("\n" + "="*60)
    print("📊 RAPPORT D'ÉVALUATION DU MODÈLE")
    print("="*60)
    
    # Informations générales
    print(f"\n🕐 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Source données: {test_data_result.get('data_source', 'N/A')}")
    print(f"🔬 Méthode évaluation: {eval_result.get('evaluation_method', 'N/A')}")
    
    # Métriques
    print(f"\n📈 MÉTRIQUES:")
    print(f"  - Précision: {eval_result.get('accuracy', 0):.2%}")
    print(f"  - Tests réussis: {eval_result.get('successful_tests', 0)}/{eval_result.get('total_tests', 0)}")
    print(f"  - Taux de réussite: {eval_result.get('success_rate', 0):.2%}")
    
    # Statut
    print(f"\n🎯 STATUT: {performance_result.get('status', 'UNKNOWN')}")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    recommendations = performance_result.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Détails des prédictions (échantillon)
    predictions = eval_result.get('predictions', [])
    if predictions:
        print(f"\n🔍 ÉCHANTILLON DE PRÉDICTIONS:")
        for pred in predictions[:5]:  # Afficher les 5 premiers
            status = "✅" if pred['correct'] else "❌"
            print(f"  {status} {pred['true_label']} -> {pred['predicted_label']} ({pred['confidence']:.2%})")
    
    print("\n" + "="*60)
    
    return {
        'report_generated': True,
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'accuracy': eval_result.get('accuracy', 0),
            'status': performance_result.get('status', 'UNKNOWN'),
            'needs_action': performance_result.get('needs_retraining', False)
        }
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
    'model_evaluation_pipeline',
    default_args=default_args,
    description='Pipeline d\'évaluation des performances du modèle',
    schedule_interval=timedelta(hours=6),  # Toutes les 6 heures
    catchup=False,
    tags=['ml', 'evaluation', 'monitoring', 'performance'],
    max_active_runs=1,
    doc_md="""
    ## Pipeline d'évaluation des performances
    
    Ce DAG évalue régulièrement les performances du modèle en production.
    
    ### Fonctionnalités:
    - Évaluation via API ou locale (fallback)
    - Tests sur données réelles ou URLs par défaut
    - Génération de rapports détaillés
    - Recommandations automatiques
    
    ### Seuils de performance:
    - Précision minimale: 70%
    - Taux de réussite: 80%
    """
)

# Définition des tâches
load_test_data_task = PythonOperator(
    task_id='load_test_data',
    python_callable=load_test_data,
    provide_context=True,
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_current_model',
    python_callable=evaluate_current_model,
    provide_context=True,
    dag=dag
)

check_performance_task = PythonOperator(
    task_id='check_model_performance',
    python_callable=check_model_performance,
    provide_context=True,
    dag=dag
)

generate_report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    provide_context=True,
    dag=dag
)

# Définir les dépendances
load_test_data_task >> evaluate_model_task >> check_performance_task >> generate_report_task