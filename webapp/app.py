import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import os
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Plant Classification App",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

def check_api_health():
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_image(image_file):
    """Envoie une image à l'API pour prédiction"""
    try:
        files = {"file": image_file}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None

def predict_from_url(image_url):
    try:
        # Utiliser le nouveau endpoint avec body JSON
        response = requests.post(
            f"{API_URL}/predict-url",
            json={"image_url": image_url},  # Envoyer en JSON
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None

def display_prediction_results(result):
    """Affiche les résultats de prédiction"""
    if result:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Résultat de la prédiction")
            
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            
            # Emoji selon la classe
            emoji = "🌼" if predicted_class == "dandelion" else "🌿"
            
            st.success(f"{emoji} **{predicted_class.title()}** ({confidence:.2%} de confiance)")
            
            # Barre de progression
            st.progress(confidence)
            
        with col2:
            st.subheader("📊 Probabilités détaillées")
            
            probs = result["probabilities"]
            
            # Graphique en barres
            fig = px.bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                labels={'x': 'Classe', 'y': 'Probabilité'},
                title="Probabilités par classe",
                color=list(probs.values()),
                color_continuous_scale="viridis"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        # Détails techniques
        with st.expander("🔍 Détails techniques"):
            st.json(result)

def main():
    """Fonction principale de l'application"""
    
    # Titre principal
    st.title("🌱 Plant Classification App")
    st.markdown("**Classifiez vos images de plantes: Pissenlit vs Herbe**")
    
    # Vérification de l'API
    if not check_api_health():
        st.error("❌ L'API n'est pas disponible. Vérifiez que le service API est démarré.")
        st.stop()
    else:
        st.success("✅ API connectée et fonctionnelle")
    
    # Sidebar
    st.sidebar.header("🎛️ Options")
    
    # Mode de prédiction
    prediction_mode = st.sidebar.selectbox(
        "Mode de prédiction",
        ["Upload d'image", "URL d'image", "Images d'exemple"]
    )
    
    # Contenu principal
    if prediction_mode == "Upload d'image":
        st.header("📁 Upload d'une image")
        
        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="Formats supportés: JPG, JPEG, PNG, BMP, GIF"
        )
        
        if uploaded_file is not None:
            # Afficher l'image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🖼️ Image uploadée")
                image = Image.open(uploaded_file)
                st.image(image, caption="Image à classifier", use_column_width=True)
            
            with col2:
                st.subheader("🔮 Prédiction")
                
                if st.button("🚀 Classifier l'image", type="primary"):
                    with st.spinner("Classification en cours..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        result = predict_image(uploaded_file)
                        
                        if result:
                            display_prediction_results(result)
    
    elif prediction_mode == "URL d'image":
        st.header("🔗 Prédiction depuis une URL")
        
        image_url = st.text_input(
            "URL de l'image",
            placeholder="https://example.com/image.jpg",
            help="Entrez l'URL complète de l'image"
        )
        
        if image_url:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🖼️ Image depuis URL")
                try:
                    st.image(image_url, caption="Image à classifier", use_column_width=True)
                except:
                    st.error("Impossible d'afficher l'image depuis cette URL")
            
            with col2:
                st.subheader("🔮 Prédiction")
                
                if st.button("🚀 Classifier l'image", type="primary"):
                    with st.spinner("Classification en cours..."):
                        result = predict_from_url(image_url)
                        
                        if result:
                            display_prediction_results(result)
    
    elif prediction_mode == "Images d'exemple":
        st.header("🎨 Images d'exemple")
        
        # URLs d'exemple
        example_images = {
            "Pissenlit 1": "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg",
            "Pissenlit 2": "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000001.jpg",
            "Herbe 1": "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000000.jpg",
            "Herbe 2": "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000001.jpg",
        }
        
        # Sélecteur d'exemple
        selected_example = st.selectbox(
            "Choisissez une image d'exemple",
            list(example_images.keys())
        )
        
        if selected_example:
            selected_url = example_images[selected_example]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"🖼️ {selected_example}")
                st.image(selected_url, caption=selected_example, use_column_width=True)
            
            with col2:
                st.subheader("🔮 Prédiction")
                
                if st.button("🚀 Classifier l'image", type="primary"):
                    with st.spinner("Classification en cours..."):
                        result = predict_from_url(selected_url)
                        
                        if result:
                            display_prediction_results(result)
    
    # Sidebar - Informations
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Informations")
    
    # Informations sur l'API
    if st.sidebar.button("🔄 Rafraîchir les infos API"):
        try:
            response = requests.get(f"{API_URL}/model-info", timeout=5)
            if response.status_code == 200:
                model_info = response.json()
                st.sidebar.success("✅ Modèle chargé")
                st.sidebar.json(model_info)
        except:
            st.sidebar.error("❌ Impossible de récupérer les infos")
    
    # Statistiques d'utilisation (simulation)
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Statistiques")
    
    # Simulation de statistiques
    if st.sidebar.button("📊 Voir les stats"):
        # Données d'exemple
        stats_data = {
            "Prédictions aujourd'hui": 42,
            "Précision du modèle": "87.5%",
            "Temps moyen de réponse": "1.2s",
            "Dernière mise à jour": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        for key, value in stats_data.items():
            st.sidebar.metric(key, value)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🌱 Plant Classification App - Projet MLOps</p>
            <p>Développé avec ❤️ en utilisant Streamlit et FastAPI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
