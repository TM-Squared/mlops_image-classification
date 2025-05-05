-- This script creates a database and a table to store metadata about the images.
CREATE DATABASE IF NOT EXISTS plants;
USE plants;
-- Create a table to store metadata about the images
CREATE TABLE IF NOT EXISTS plants_data (
    url_source TEXT NOT NULL,
    url_s3 TEXT,
    label ENUM('dandelion', 'grass') NOT NULL
);

-- The following line is commented out to avoid loading the dandelion data again
LOAD DATA INFILE '/docker-entrypoint-initdb.d/dandelions_clean.csv'
INTO TABLE plants_data
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(url_source)
SET label = 'dandelion';

-- The following line is commented out to avoid loading the grass data again
LOAD DATA INFILE '/docker-entrypoint-initdb.d/grass_clean.csv'
INTO TABLE plants_data
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(url_source)
SET label = 'grass';