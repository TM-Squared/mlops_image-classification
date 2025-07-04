from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
import os
import requests
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
import tempfile
import json

# Configuration TensorFlow
tf.config.set_visible_devices([], 'GPU')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageUrlRequest(BaseModel):
    image_url: str

app = FastAPI(
    title="Plant Classification API (TensorFlow + MinIO)",
    description="API pour la classification d'images de plantes avec TensorFlow et stockage MinIO",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
class_names = {0: "grass", 1: "dandelion"}
minio_client = None

class MinIOModelManager:
    """Gestionnaire pour charger des modèles depuis MinIO"""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
            region_name='us-east-1'
        )
        self.bucket_name = 'models'
    
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
                logger.error(f"Erreur listage modèles: {e}")
        
        # Essayer de charger chaque modèle
        for s3_key in possible_keys:
            try:
                logger.info(f"Tentative de chargement: s3://{self.bucket_name}/{s3_key}")
                
                # Vérifier si l'objet existe
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                
                # Télécharger dans un fichier temporaire
                with tempfile.NamedTemporaryFile(suffix='.keras' if s3_key.endswith('.keras') else '.h5', delete=False) as tmp_file:
                    self.s3_client.download_file(self.bucket_name, s3_key, tmp_file.name)
                    
                    # Charger le modèle
                    model = keras.models.load_model(tmp_file.name)
                    
                    # Nettoyer le fichier temporaire
                    os.unlink(tmp_file.name)
                    
                    logger.info(f"Modèle chargé depuis MinIO: s3://{self.bucket_name}/{s3_key}")
                    return model, s3_key
                    
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.warning(f"Modèle non trouvé: s3://{self.bucket_name}/{s3_key}")
                else:
                    logger.error(f"Erreur accès modèle {s3_key}: {e}")
            except Exception as e:
                logger.error(f"Erreur chargement modèle {s3_key}: {e}")
        
        logger.error("Aucun modèle trouvé dans MinIO")
        return None, None
    
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
            logger.error(f"Erreur listage modèles: {e}")
            return []

def preprocess_image(image):
    """Preprocessing d'une image PIL"""
    try:
        # Redimensionner
        image = image.resize((224, 224))
        
        # Convertir en array et normaliser
        image_array = np.array(image) / 255.0
        
        # Ajouter la dimension batch
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Erreur preprocessing: {e}")
        raise

def load_model():
    """Charge le modèle TensorFlow depuis MinIO"""
    global model, minio_client
    
    minio_client = MinIOModelManager()
    
    # Essayer de charger le modèle depuis MinIO
    try:
        loaded_model, model_key = minio_client.load_model_from_minio("plant_classifier", "latest")
        
        if loaded_model:
            model = loaded_model
            logger.info(f"✅ Modèle chargé depuis MinIO: {model_key}")
            return True
            
    except Exception as e:
        logger.error(f"Erreur chargement modèle MinIO: {e}")
    
    # Fallback: créer un modèle par défaut
    logger.warning("Création d'un modèle par défaut")
    model = create_default_model()
    logger.info("Modèle par défaut créé")
    return False

def create_default_model():
    """Crée un modèle par défaut pour les tests"""
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Ajouter cette fonction avant les endpoints
def validate_image_file(file: UploadFile) -> bool:
    """Valide qu'un fichier est bien une image"""
    
    # Vérifier le content-type si disponible
    if file.content_type:
        valid_content_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 
            'image/gif', 'image/webp', 'image/tiff'
        ]
        if file.content_type not in valid_content_types:
            return False
    
    # Vérifier l'extension du fichier
    if file.filename:
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']
        filename_lower = file.filename.lower()
        if not any(filename_lower.endswith(ext) for ext in valid_extensions):
            return False
    
    return True

def get_image_info(image_bytes: bytes) -> dict:
    """Obtenir des informations sur l'image"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height
        }
    except Exception as e:
        logger.error(f"Erreur analyse image: {e}")
        return {}

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Démarrage de l'API Plant Classification (TensorFlow + MinIO)")
    
    # Créer les dossiers nécessaires
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Charger le modèle
    model_loaded = load_model()
    
    if model_loaded:
        logger.info("✅ API prête avec modèle MinIO")
    else:
        logger.info("✅ API prête avec modèle par défaut")

@app.get("/")
async def root():
    return {
        "message": "Plant Classification API (TensorFlow + MinIO)",
        "version": "1.0.0",
        "framework": "TensorFlow",
        "storage": "MinIO",
        "tf_version": tf.__version__,
        "status": "running",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "framework": "TensorFlow",
        "storage": "MinIO",
        "tf_version": tf.__version__,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prédiction sur une image uploadée"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Validation du fichier
    if not validate_image_file(file):
        raise HTTPException(
            status_code=400, 
            detail=f"Fichier non valide. Formats acceptés: JPG, JPEG, PNG, BMP, GIF, WEBP. "
                   f"Reçu: {file.content_type} - {file.filename}"
        )
    
    try:
        # Lire l'image
        image_bytes = await file.read()
        
        # Vérifier que le fichier n'est pas vide
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Le fichier est vide")
        
        # Vérifier la taille du fichier (max 10MB)
        max_file_size = 10 * 1024 * 1024  # 10 MB
        if len(image_bytes) > max_file_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Fichier trop volumineux ({len(image_bytes)} bytes). Maximum: {max_file_size} bytes"
            )
        
        # Obtenir les informations sur l'image
        image_info = get_image_info(image_bytes)
        
        # Tenter d'ouvrir l'image pour vérifier qu'elle est valide
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            logger.info(f"Image chargée: {image_info}")
        except Exception as img_error:
            raise HTTPException(
                status_code=400, 
                detail=f"Impossible d'ouvrir l'image: {str(img_error)}"
            )
        
        # Vérifier les dimensions minimales
        min_size = 32
        if image.width < min_size or image.height < min_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Image trop petite ({image.width}x{image.height}). Minimum: {min_size}x{min_size}"
            )
        
        # Preprocessing
        image_array = preprocess_image(image)
        
        # Prédiction
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_label = class_names[predicted_class_idx]
        
        # Probabilités pour chaque classe
        prob_grass = float(predictions[0][0])
        prob_dandelion = float(predictions[0][1])
        
        result = {
            "predicted_class": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "grass": round(prob_grass, 4),
                "dandelion": round(prob_dandelion, 4)
            },
            "framework": "TensorFlow",
            "storage": "MinIO",
            "tf_version": tf.__version__,
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": len(image_bytes),
                **image_info
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prédiction: {predicted_label} ({confidence:.2%}) - {file.filename}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        # Re-lever les HTTPException telles quelles
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")


@app.post("/predict-url")
async def predict_from_url(request: ImageUrlRequest):
    """Prédiction depuis une URL d'image (POST avec body JSON)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Télécharger l'image
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Preprocessing
        image_array = preprocess_image(image)
        
        # Prédiction
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_label = class_names[predicted_class_idx]
        
        # Probabilités pour chaque classe
        prob_grass = float(predictions[0][0])
        prob_dandelion = float(predictions[0][1])
        
        result = {
            "predicted_class": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "grass": round(prob_grass, 4),
                "dandelion": round(prob_dandelion, 4)
            },
            "image_url": request.image_url,
            "framework": "TensorFlow",
            "storage": "MinIO",
            "tf_version": tf.__version__,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prédiction URL: {predicted_label} ({confidence:.2%})")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction depuis URL: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/predict-url-get")
async def predict_from_url_get(image_url: str):
    """Prédiction depuis une URL d'image (GET avec query parameter)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Télécharger l'image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Preprocessing
        image_array = preprocess_image(image)
        
        # Prédiction
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_label = class_names[predicted_class_idx]
        
        # Probabilités pour chaque classe
        prob_grass = float(predictions[0][0])
        prob_dandelion = float(predictions[0][1])
        
        result = {
            "predicted_class": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "grass": round(prob_grass, 4),
                "dandelion": round(prob_dandelion, 4)
            },
            "image_url": image_url,
            "framework": "TensorFlow",
            "storage": "MinIO",
            "tf_version": tf.__version__,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prédiction URL GET: {predicted_label} ({confidence:.2%})")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction depuis URL: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.post("/reload-model")
async def reload_model():
    try:
        model_loaded = load_model()
        
        if model_loaded:
            message = "Modèle rechargé depuis MinIO avec succès"
            logger.info(message)
        else:
            message = "Modèle par défaut rechargé (MinIO non accessible)"
            logger.warning(message)
        
        return {
            "message": message,
            "framework": "TensorFlow",
            "storage": "MinIO",
            "tf_version": tf.__version__,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erreur lors du rechargement du modèle: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/models")
async def list_models():
    """Lister tous les modèles disponibles dans MinIO"""
    try:
        if minio_client:
            models = minio_client.list_models("plant_classifier")
            return {
                "available_models": models,
                "total_models": len(models),
                "storage": "MinIO",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "available_models": [],
                "total_models": 0,
                "storage": "MinIO",
                "error": "Client MinIO non initialisé",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Erreur listage modèles: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/model-info")
async def model_info():
    return {
        "model_loaded": model is not None,
        "framework": "TensorFlow",
        "storage": "MinIO",
        "tf_version": tf.__version__,
        "model_type": "MobileNetV2",
        "classes": list(class_names.values()),
        "input_size": [224, 224, 3],
        "supported_formats": [".keras", ".h5"],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)