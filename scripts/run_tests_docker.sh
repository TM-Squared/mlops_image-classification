#!/bin/bash

echo "🧪 Lancement des tests dans Docker"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Nettoyer les anciens conteneurs
print_status "Nettoyage des anciens conteneurs de test..."
docker compose -f docker-compose.test.yml down -v

# Construire les images
print_status "Construction des images de test..."
docker compose -f docker-compose.test.yml build

# Démarrer les services
print_status "Démarrage des services de test..."
docker compose -f docker-compose.test.yml up -d postgres mysql minio mlflow api

# Attendre que les services soient prêts
print_status "Attente de la disponibilité des services..."
sleep 45

# Vérifier la santé des services
print_status "Vérification de la santé des services..."

services=(
    "postgres:5432"
    "mysql:3306"
    "minio:9000"
    "mlflow:5000"
    "api:8000"
)

for service in "${services[@]}"; do
    IFS=':' read -r host port <<< "$service"
    if docker compose -f docker-compose.test.yml exec -T test-runner nc -z $host $port; then
        print_success "Service $service disponible"
    else
        print_error "Service $service non disponible"
    fi
done

# Lancer les tests
print_status "Lancement des tests..."
docker compose -f docker-compose.test.yml run --rm test-runner

# Capturer le code de sortie
test_exit_code=$?

# Copier les résultats de test
print_status "Copie des résultats de test..."
docker cp $(docker compose -f docker-compose.test.yml ps -q test-runner):/app/test-results ./test-results 2>/dev/null || true

# Nettoyer
print_status "Nettoyage..."
docker compose -f docker-compose.test.yml down -v

# Résultats
if [ $test_exit_code -eq 0 ]; then
    print_success "Tous les tests sont passés!"
    echo ""
    echo "📊 Résultats disponibles dans:"
    echo "   - ./test-results/junit.xml"
    echo "   - ./test-results/report.html"
else
    print_error "Certains tests ont échoué"
fi

exit $test_exit_code