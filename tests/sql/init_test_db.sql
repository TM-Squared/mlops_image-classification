-- Base de données de test
CREATE DATABASE IF NOT EXISTS plants_test;
USE plants_test;

-- Table pour les données de plantes
CREATE TABLE IF NOT EXISTS plants_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url_source VARCHAR(500) NOT NULL,
    url_s3 VARCHAR(500),
    label VARCHAR(50) NOT NULL,
    image_exists BOOLEAN DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insérer des données de test
INSERT INTO plants_data (url_source, url_s3, label, image_exists) VALUES
('https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg', 's3://raw-data/raw/dandelion/00000000.jpg', 'dandelion', TRUE),
('https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000001.jpg', 's3://raw-data/raw/dandelion/00000001.jpg', 'dandelion', TRUE),
('https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000000.jpg', 's3://raw-data/raw/grass/00000000.jpg', 'grass', TRUE),
('https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000001.jpg', 's3://raw-data/raw/grass/00000001.jpg', 'grass', TRUE);

-- Base de données principale
USE plants;

-- Table pour les données de plantes
CREATE TABLE IF NOT EXISTS plants_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url_source VARCHAR(500) NOT NULL,
    url_s3 VARCHAR(500),
    label VARCHAR(50) NOT NULL,
    image_exists BOOLEAN DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
