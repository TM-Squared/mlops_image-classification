import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import mlflow
import mlflow.tensorflow
from PIL import Image
import io
import boto3
import os
import tempfile
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from botocore.exceptions import ClientError, NoCredentialsError

# Configuration pour éviter les erreurs GPU
tf.config.set_visible_devices([], 'GPU')

class MinIOModelManager:
    """Gestionnaire pour sauvegarder/charger des modèles depuis MinIO"""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
            region_name='us-east-1'
        )
        self.bucket_name = 'models'
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Assurer que le bucket existe"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    print(f"✅ Bucket {self.bucket_name} créé")
                except Exception as create_error:
                    print(f"⚠️ Erreur création bucket: {create_error}")
            else:
                print(f"⚠️ Erreur accès bucket: {e}")
    
    def save_model_to_minio(self, model, model_name="plant_classifier"):
        """Sauvegarder un modèle TensorFlow sur MinIO"""
        saved_keys = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Sauvegarder en format .keras
            try:
                keras_path = os.path.join(tmp_dir, f"{model_name}.keras")
                model.save(keras_path)
                
                # Upload vers MinIO
                s3_key = f"tensorflow/{model_name}_{timestamp}.keras"
                self.s3_client.upload_file(keras_path, self.bucket_name, s3_key)
                saved_keys.append(s3_key)
                print(f"✅ Modèle Keras uploadé: s3://{self.bucket_name}/{s3_key}")
                
                # Sauvegarder aussi la version "latest"
                latest_key = f"tensorflow/{model_name}_latest.keras"
                self.s3_client.upload_file(keras_path, self.bucket_name, latest_key)
                saved_keys.append(latest_key)
                print(f"✅ Modèle Keras latest: s3://{self.bucket_name}/{latest_key}")
                
            except Exception as e:
                print(f"⚠️ Erreur sauvegarde Keras: {e}")
            
            # Sauvegarder en format .h5
            try:
                h5_path = os.path.join(tmp_dir, f"{model_name}.h5")
                model.save(h5_path)
                
                # Upload vers MinIO
                s3_key = f"tensorflow/{model_name}_{timestamp}.h5"
                self.s3_client.upload_file(h5_path, self.bucket_name, s3_key)
                saved_keys.append(s3_key)
                print(f"✅ Modèle H5 uploadé: s3://{self.bucket_name}/{s3_key}")
                
                # Sauvegarder aussi la version "latest"
                latest_key = f"tensorflow/{model_name}_latest.h5"
                self.s3_client.upload_file(h5_path, self.bucket_name, latest_key)
                saved_keys.append(latest_key)
                print(f"✅ Modèle H5 latest: s3://{self.bucket_name}/{latest_key}")
                
            except Exception as e:
                print(f"⚠️ Erreur sauvegarde H5: {e}")
            
            # Sauvegarder les métadonnées
            try:
                metadata = {
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "tf_version": tf.__version__,
                    "model_type": "MobileNetV2",
                    "input_shape": [224, 224, 3],
                    "num_classes": 2,
                    "class_names": {0: "grass", 1: "dandelion"}
                }
                
                metadata_path = os.path.join(tmp_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                metadata_key = f"tensorflow/{model_name}_{timestamp}_metadata.json"
                self.s3_client.upload_file(metadata_path, self.bucket_name, metadata_key)
                saved_keys.append(metadata_key)
                print(f"✅ Métadonnées uploadées: s3://{self.bucket_name}/{metadata_key}")
                
            except Exception as e:
                print(f"⚠️ Erreur sauvegarde métadonnées: {e}")
        
        if not saved_keys:
            raise Exception("❌ Aucune sauvegarde MinIO n'a fonctionné")
        
        return saved_keys
    
    def load_model_from_minio(self, model_name="plant_classifier", version="latest"):
        """Charger un modèle depuis MinIO"""
        
        # Essayer différents formats
        possible_keys = [
            f"tensorflow/{model_name}_{version}.keras",
            f"tensorflow/{model_name}_{version}.h5",
            f"tensorflow/{model_name}_latest.keras",
            f"tensorflow/{model_name}_latest.h5"
        ]
        
        # Lister tous les modèles disponibles si version n'est pas "latest"
        if version != "latest":
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"tensorflow/{model_name}_"
                )
                
                if 'Contents' in response:
                    # Trier par date de modification (plus récent en premier)
                    objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    
                    # Ajouter les modèles les plus récents à la liste
                    for obj in objects[:3]:  # Prendre les 3 plus récents
                        key = obj['Key']
                        if key.endswith(('.keras', '.h5')) and key not in possible_keys:
                            possible_keys.append(key)
                            
            except Exception as e:
                print(f"⚠️ Erreur listage modèles: {e}")
        
        # Essayer de charger chaque modèle
        for s3_key in possible_keys:
            try:
                print(f"🔍 Tentative de chargement: s3://{self.bucket_name}/{s3_key}")
                
                # Vérifier si l'objet existe
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                
                # Télécharger dans un fichier temporaire
                with tempfile.NamedTemporaryFile(suffix='.keras' if s3_key.endswith('.keras') else '.h5', delete=False) as tmp_file:
                    self.s3_client.download_file(self.bucket_name, s3_key, tmp_file.name)
                    
                    # Charger le modèle
                    model = keras.models.load_model(tmp_file.name)
                    
                    # Nettoyer le fichier temporaire
                    os.unlink(tmp_file.name)
                    
                    print(f"✅ Modèle chargé depuis MinIO: s3://{self.bucket_name}/{s3_key}")
                    return model
                    
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print(f"⚠️ Modèle non trouvé: s3://{self.bucket_name}/{s3_key}")
                else:
                    print(f"⚠️ Erreur accès modèle {s3_key}: {e}")
            except Exception as e:
                print(f"⚠️ Erreur chargement modèle {s3_key}: {e}")
        
        print("❌ Aucun modèle trouvé dans MinIO")
        return None
    
    def list_models(self, model_name="plant_classifier"):
        """Lister tous les modèles disponibles"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"tensorflow/{model_name}_"
            )
            
            if 'Contents' in response:
                models = []
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith(('.keras', '.h5')):
                        models.append({
                            'key': key,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'format': 'keras' if key.endswith('.keras') else 'h5'
                        })
                
                # Trier par date de modification (plus récent en premier)
                models.sort(key=lambda x: x['last_modified'], reverse=True)
                return models
            else:
                return []
                
        except Exception as e:
            print(f"⚠️ Erreur listage modèles: {e}")
            return []

class MinIOImageDataGenerator(keras.utils.Sequence):
    def __init__(self, s3_keys, labels, batch_size=8, img_size=(224, 224), shuffle=True):
        self.s3_keys = s3_keys
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.s3_keys))
        
        # Initialiser le client S3/MinIO
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
            region_name='us-east-1'
        )
        
        self.bucket_name = 'raw-data'
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.s3_keys) // self.batch_size
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        
        for i in batch_indices:
            try:
                # Récupérer l'image depuis MinIO
                s3_key = self.s3_keys[i]
                
                # Télécharger l'image depuis MinIO
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                image_data = response['Body'].read()
                
                # Ouvrir l'image
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Redimensionner et normaliser
                image = image.resize(self.img_size)
                image_array = np.array(image) / 255.0
                
                batch_images.append(image_array)
                
                # Label: 0 = grass, 1 = dandelion
                label = 1 if self.labels[i] == 'dandelion' else 0
                batch_labels.append(label)
                
            except Exception as e:
                print(f"Erreur chargement image {i} ({s3_key}): {e}")
                # Image par défaut
                default_image = np.random.random((*self.img_size, 3))
                batch_images.append(default_image)
                batch_labels.append(0)
        
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Générateur simple pour les URLs (fallback)
class SimpleImageDataGenerator(keras.utils.Sequence):
    def __init__(self, image_urls, labels, batch_size=8, img_size=(224, 224), shuffle=True):
        self.image_urls = image_urls
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_urls))
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.image_urls) // self.batch_size
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        
        for i in batch_indices:
            try:
                # Télécharger l'image
                import requests
                response = requests.get(self.image_urls[i], timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                
                # Redimensionner et normaliser
                image = image.resize(self.img_size)
                image_array = np.array(image) / 255.0
                
                batch_images.append(image_array)
                
                # Label: 0 = grass, 1 = dandelion
                label = 1 if self.labels[i] == 'dandelion' else 0
                batch_labels.append(label)
                
            except Exception as e:
                print(f"Erreur chargement image {i}: {e}")
                # Image par défaut
                default_image = np.random.random((*self.img_size, 3))
                batch_images.append(default_image)
                batch_labels.append(0)
        
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_simple_model(input_shape=(224, 224, 3), num_classes=2):
    """Crée un modèle simple avec MobileNetV2"""
    
    # Base model avec MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Geler les couches de base
    base_model.trainable = False
    
    # Ajouter les couches personnalisées
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model_from_minio(s3_keys, labels, num_epochs=3):
    """Entraîne le modèle avec les données depuis MinIO"""
    print(f"Entraînement avec {len(s3_keys)} images depuis MinIO")
    
    # Diviser les données
    train_keys, val_keys, train_labels, val_labels = train_test_split(
        s3_keys, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(train_keys)}, Val: {len(val_keys)}")
    
    # Créer les générateurs
    train_generator = MinIOImageDataGenerator(
        train_keys, train_labels, batch_size=8, shuffle=True
    )
    val_generator = MinIOImageDataGenerator(
        val_keys, val_labels, batch_size=8, shuffle=False
    )
    
    # Créer le modèle
    model = create_simple_model()
    
    # Compiler le modèle
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Architecture du modèle:")
    model.summary()
    
    # MLflow tracking
    mlflow.set_experiment("plant-classification-minio")
    
    with mlflow.start_run():
        # Log des paramètres
        mlflow.log_params({
            "model_type": "MobileNetV2_TensorFlow",
            "data_source": "MinIO",
            "num_epochs": num_epochs,
            "batch_size": 8,
            "learning_rate": 0.001,
            "train_samples": len(train_keys),
            "val_samples": len(val_keys),
            "optimizer": "Adam",
            "base_model": "MobileNetV2",
            "tf_version": tf.__version__
        })
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=2,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1,
                min_lr=1e-7
            )
        ]
        
        # Entraînement
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=num_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation finale
        val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
        
        print(f"\nRésultats finaux:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Log des métriques finales
        mlflow.log_metrics({
            "final_val_loss": val_loss,
            "final_val_accuracy": val_accuracy,
            "best_val_accuracy": max(history.history['val_accuracy'])
        })
        
        # Sauvegarder le modèle sur MinIO
        minio_manager = MinIOModelManager()
        try:
            saved_keys = minio_manager.save_model_to_minio(model, "plant_classifier")
            print(f"✅ Modèle sauvegardé sur MinIO: {saved_keys}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde MinIO: {e}")
            saved_keys = []
        
        # Log du modèle dans MLflow
        try:
            mlflow.tensorflow.log_model(
                model,
                "model",
                registered_model_name="plant-classifier-minio"
            )
            print("✅ Modèle enregistré dans MLflow")
        except Exception as e:
            print(f"⚠️ Erreur enregistrement MLflow: {e}")
        
        print(f"Modèle sauvegardé sur MinIO avec les clés: {saved_keys}")
        
        return model, val_accuracy

def train_quick_model(image_urls, labels, num_epochs=3):
    """Entraîne le modèle avec des URLs (fallback)"""
    print(f"Entraînement avec {len(image_urls)} images depuis URLs")
    
    # Diviser les données
    train_urls, val_urls, train_labels, val_labels = train_test_split(
        image_urls, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(train_urls)}, Val: {len(val_urls)}")
    
    # Créer les générateurs
    train_generator = SimpleImageDataGenerator(
        train_urls, train_labels, batch_size=8, shuffle=True
    )
    val_generator = SimpleImageDataGenerator(
        val_urls, val_labels, batch_size=8, shuffle=False
    )
    
    # Créer le modèle
    model = create_simple_model()
    
    # Compiler le modèle
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Architecture du modèle:")
    model.summary()
    
    # MLflow tracking
    mlflow.set_experiment("plant-classification-quick")
    
    with mlflow.start_run():
        # Log des paramètres
        mlflow.log_params({
            "model_type": "MobileNetV2_TensorFlow",
            "data_source": "URLs",
            "num_epochs": num_epochs,
            "batch_size": 8,
            "learning_rate": 0.001,
            "train_samples": len(train_urls),
            "val_samples": len(val_urls),
            "optimizer": "Adam",
            "base_model": "MobileNetV2",
            "tf_version": tf.__version__
        })
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=2,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1,
                min_lr=1e-7
            )
        ]
        
        # Entraînement
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=num_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation finale
        val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
        
        print(f"\nRésultats finaux:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Log des métriques finales
        mlflow.log_metrics({
            "final_val_loss": val_loss,
            "final_val_accuracy": val_accuracy,
            "best_val_accuracy": max(history.history['val_accuracy'])
        })
        
        # Sauvegarder le modèle sur MinIO
        minio_manager = MinIOModelManager()
        try:
            saved_keys = minio_manager.save_model_to_minio(model, "plant_classifier")
            print(f"✅ Modèle sauvegardé sur MinIO: {saved_keys}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde MinIO: {e}")
            saved_keys = []
        
        # Log du modèle dans MLflow
        try:
            mlflow.tensorflow.log_model(
                model,
                "model",
                registered_model_name="plant-classifier-quick"
            )
            print("✅ Modèle enregistré dans MLflow")
        except Exception as e:
            print(f"⚠️ Erreur enregistrement MLflow: {e}")
        
        print(f"Modèle sauvegardé sur MinIO avec les clés: {saved_keys}")
        
        return model, val_accuracy

def load_model_for_prediction(model_name="plant_classifier", version="latest"):
    """Charge le modèle pour les prédictions depuis MinIO"""
    minio_manager = MinIOModelManager()
    model = minio_manager.load_model_from_minio(model_name, version)
    return model

def predict_image_from_minio(model, s3_key):
    """Fait une prédiction sur une image depuis MinIO"""
    try:
        # Initialiser le client S3/MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
            region_name='us-east-1'
        )
        
        # Télécharger l'image depuis MinIO
        response = s3_client.get_object(Bucket='raw-data', Key=s3_key)
        image_data = response['Body'].read()
        
        # Ouvrir l'image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocessing
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Prédiction
        predictions = model.predict(image_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        class_names = {0: "grass", 1: "dandelion"}
        
        return {
            "predicted_class": class_names[predicted_class],
            "confidence": float(confidence),
            "probabilities": {
                "grass": float(predictions[0][0]),
                "dandelion": float(predictions[0][1])
            }
        }
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None

def predict_image_from_url(model, image_url):
    """Fait une prédiction sur une image depuis une URL"""
    try:
        import requests
        
        # Télécharger l'image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Preprocessing
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Prédiction
        predictions = model.predict(image_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        class_names = {0: "grass", 1: "dandelion"}
        
        return {
            "predicted_class": class_names[predicted_class],
            "confidence": float(confidence),
            "probabilities": {
                "grass": float(predictions[0][0]),
                "dandelion": float(predictions[0][1])
            }
        }
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None

if __name__ == "__main__":
    # Test avec des URLs
    print("Test du modèle TensorFlow avec sauvegarde MinIO...")
    
    # URLs de test
    test_urls = [
        "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg",
        "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000001.jpg",
        "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000000.jpg",
        "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000001.jpg",
    ]
    
    test_labels = ['dandelion', 'dandelion', 'grass', 'grass']
    
    # Entraînement rapide
    model, accuracy = train_quick_model(test_urls, test_labels, num_epochs=2)
    print(f"Modèle entraîné avec une précision de {accuracy:.2%}")
    
    # Test de chargement depuis MinIO
    print("\nTest de chargement depuis MinIO:")
    loaded_model = load_model_for_prediction()
    if loaded_model:
        print("✅ Modèle chargé depuis MinIO")
        
        # Test de prédiction
        result = predict_image_from_url(loaded_model, test_urls[0])
        if result:
            print(f"Prédiction: {result}")
    else:
        print("❌ Impossible de charger le modèle depuis MinIO")