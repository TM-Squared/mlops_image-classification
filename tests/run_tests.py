import subprocess
import sys
import os

def run_command(command, description):
    """Exécuter une commande et afficher le résultat"""
    print(f"\n{'='*50}")
    print(f"🧪 {description}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCÈS")
        if result.stdout:
            print("📋 Sortie:")
            print(result.stdout)
    else:
        print(f"❌ {description} - ÉCHEC")
        if result.stderr:
            print("🚨 Erreur:")
            print(result.stderr)
        if result.stdout:
            print("📋 Sortie:")
            print(result.stdout)
    
    return result.returncode == 0

def main():
    """Fonction principale"""
    print("🚀 Lancement de la suite de tests complète")
    
    # Changer vers le répertoire du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    tests_commands = [
        ("python -m pytest tests/test_models.py -v", "Tests unitaires - Modèles"),
        ("python -m pytest tests/test_api.py -v", "Tests unitaires - API"),
        ("python -m pytest tests/test_training.py -v", "Tests unitaires - Entraînement"),
        ("python -m pytest tests/test_airflow_dags.py -v", "Tests unitaires - DAGs Airflow"),
        ("python -m pytest tests/test_integration.py -v", "Tests d'intégration"),
        ("python -m pytest tests/test_performance.py -v", "Tests de performance"),
        ("python -m pytest tests/ --cov=ml --cov=api --cov-report=html", "Tests avec couverture"),
    ]
    
    results = []
    
    for command, description in tests_commands:
        success = run_command(command, description)
        results.append((description, success))
    
    # Résumé final
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ DES TESTS")
    print(f"{'='*60}")
    
    for description, success in results:
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        print(f"{description:<40} {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    print(f"\n📈 Résultat global: {passed_tests}/{total_tests} tests réussis")
    
    if passed_tests == total_tests:
        print("🎉 Tous les tests sont passés!")
        return 0
    else:
        print("⚠️ Certains tests ont échoué")
        return 1

if __name__ == "__main__":
    sys.exit(main())
