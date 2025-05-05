import os
import numpy as np
import boto3
import mysql.connector
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from PIL import Image
import io

# Charger les variables d'environnement
load_dotenv()

# Config MinIO (S3-compatible)
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

label_encoder = LabelEncoder()

unique_labels = set(row['label'] for row in rows)
label_encoder.fit(list(unique_labels))

for row in rows:
    url_s3 = row['url_s3']
    label = row['label']

    # Télécharger l'image depuis MinIO
    s3_key = url_s3.split("s3://{}".format(bucket_name))[1]  # Récupérer la clé S3 (nom du fichier)
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    img_data = obj['Body'].read()

    # Convertir en image avec PIL
    img = Image.open(io.BytesIO(img_data))

    # Redimensionner l'image à 224x224 pixels (taille attendue par EfficientNet)
    img = img.resize((224, 224))

    # Convertir l'image en tableau NumPy
    img_array = np.array(img)

    # Vérification que l'image a 3 canaux (RGB), si ce n'est pas le cas, la convertir
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Prétraitement pour EfficientNet (normalisation)
    img_array = preprocess_input(img_array)  # Normalisation spécifique pour EfficientNet

    image_data.append(img_array)

    encoded_label = label_encoder.transform([label])[0]
    labels.append(encoded_label)

# Convertir en tableaux numpy
X = np.array(image_data)
y = np.array(labels)

# Créer le modèle EfficientNet
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(1, activation='sigmoid')  # Classification binaire (dandelion vs grass)
])

# Compiler le modèle
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Sauvegarder le modèle dans MinIO
model_filename = "dandelion_and_grass_classifier.h5"

# Sauvegarder le modèle sur le disque local d'abord (temporaire)
model.save(model_filename)

# Charger le modèle dans MinIO
with open(model_filename, 'rb') as model_file:
    s3.put_object(Bucket=bucket_name, Key=f'models/{model_filename}', Body=model_file)

# Suppression du fichier temporaire
os.remove(model_filename)

# Fermer la connexion à MySQL
cursor.close()
conn.close()

print(f"Model trained and saved to MinIO as 'models/{model_filename}'.")
