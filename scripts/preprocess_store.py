import os
import numpy as np
import boto3
import mysql.connector
import logging
#from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
#from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('MINIO_ENDPOINT'),
    aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('MINIO_SECRET_KEY'),
)

bucket_name = os.getenv('MINIO_BUCKET')

# Config MySQL via .env
mysql_config = {
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'host': os.getenv('MYSQL_HOST'),
    'database': os.getenv('MYSQL_DATABASE'),
}

# Connexion à MySQL
conn = mysql.connector.connect(**mysql_config)
cursor = conn.cursor(dictionary=True)

# Récupération des images à traiter
cursor.execute("SELECT url_s3, label FROM plants_data WHERE url_s3 IS NOT NULL")
rows = cursor.fetchall()

# Prétraitement des images
image_data = []
labels = []

for row in rows:
    url_s3 = row['url_s3']
    label = row['label']

    # Télécharger l'image depuis MinIO
    s3_key = url_s3.split(f"s3://{bucket_name}")[1]  # Récupérer la clé S3 (nom du fichier)
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    img_data = obj['Body'].read()

    # Convertir en image avec PIL
    img = Image.open(io.BytesIO(img_data))

    # Redimensionner l'image à 224x224 pixels (taille attendue par EfficientNetB0)
    img = img.resize((224, 224))

    # Convertir l'image en tableau NumPy
    img_array = np.array(img)

    # Vérification que l'image a 3 canaux (RGB), si ce n'est pas le cas, la convertir
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Prétraitement pour EfficientNet (normalisation)
    img_array = preprocess_input(img_array)  # Normalisation spécifique pour EfficientNet

    image_data.append(img_array)
    labels.append(label)

# Convertir en tableaux numpy
x = np.array(image_data)
y = np.array(labels)

x_file = io.BytesIO()
np.save(x_file, x)
x_file.seek(0)

s3.put_object(Bucket=bucket_name, Key='formatted/preprocessed_x.npy', Body=x_file)


y_file = io.BytesIO()
np.save(y_file, y)
y_file.seek(0)

s3.put_object(Bucket=bucket_name, Key='formatted/preprocessed_y.npy', Body=y_file)

# Fermer la connexion à MySQL
cursor.close()
conn.close()

print(f"Prétraitement terminé. {len(x)} images préparées.")
