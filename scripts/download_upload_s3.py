import os
import requests
import boto3
import mysql.connector
import logging
import uuid
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/download_upload_s3.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

mysql_config = {
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'host': os.getenv('MYSQL_HOST'),
    'database': os.getenv('MYSQL_DATABASE'),
    'allow_local_infile': True
}

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('MINIO_ENDPOINT'),
    aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('MINIO_SECRET_KEY'),
)

bucket_name = os.getenv('MINIO_BUCKET')

try:
    s3.head_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' already exists.")
except:
    s3.create_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' created.")

conn = mysql.connector.connect(**mysql_config)
cursor = conn.cursor(dictionary=True)

# Récupération des images à traiter
cursor.execute("SELECT url_source FROM plants_data WHERE url_s3 IS NULL")
rows = cursor.fetchall()
logger.info(f"{len(rows)} images to process.")

# Traitement
for row in rows:
    url = row['url_source']
    try:
        logger.info(f"Downloading: {url}")
        response = requests.get(url, timeout=10)

        if response.status_code == 404:
            # Si l'image n'est pas trouvée, on met à jour le champ `downloaded` à 0
            cursor.execute(
                "UPDATE plants_data SET downloaded = 0 WHERE url_source = %s",
                (url,)
            )
            conn.commit()
            logger.warning(f"404 Not Found for {url}. Marked as downloaded=0.")
            continue

        response.raise_for_status()  # Pour lever une exception en cas d'erreur autre que 404

        # Générer un nom de fichier unique avec UUID
        unique_filename = str(uuid.uuid4()) + ".jpg"  # Utilisation d'un UUID pour le nom de fichier
        s3_key = f"raw/{unique_filename}"

        # Upload vers MinIO
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=response.content)
        url_s3 = f"s3://{bucket_name}/{s3_key}"

        # Mise à jour de la base avec l'URL S3 et marquer comme téléchargé (1)
        cursor.execute(
            "UPDATE plants_data SET url_s3 = %s, downloaded = 1 WHERE url_source = %s",
            (url_s3, url)
        )
        conn.commit()
        logger.info(f"Uploaded and updated: {unique_filename}")

    except Exception as e:
        logger.warning(f"Failed to process {url}: {e}")
        cursor.execute(
            "UPDATE plants_data SET downloaded = 0 WHERE url_source = %s",
            (url,)
        )
        conn.commit()

# Cleanup
cursor.close()
conn.close()
logger.info("Processing completed.")
