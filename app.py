import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'svm_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model_and_scaler()

# Species mapping
species_map = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

# Species emoji
species_emoji = {
    "Iris-setosa": "🌺",
    "Iris-versicolor": "🌸",
    "Iris-virginica": "🌼"
}

# Title
st.title("🌸 Iris Flower Species Classifier")
st.markdown("### Predict iris species using machine learning!")

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    st.info("""
    **Model:** Support Vector Machine (SVM)  
    **Kernel:** Linear  
    **Accuracy:** 100% ✨  
    **Features:** 4 measurements
    """)
    
    st.header("🌺 Species Info")
    st.write("**Iris-setosa** 🌺 - Smallest petals")
    st.write("**Iris-versicolor** 🌸 - Medium features")
    st.write("**Iris-virginica** 🌼 - Largest petals")

# Stop if model not loaded
if model is None or scaler is None:
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction"])

with tab1:
    st.subheader("Enter Flower Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌿 Sepal Measurements")
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.8,
            step=0.1
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=4.5,
            value=3.0,
            step=0.1
        )
    
    with col2:
        st.markdown("#### 🌺 Petal Measurements")
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=4.3,
            step=0.1
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=2.5,
            value=1.3,
            step=0.1
        )
    
    # Display input values
    st.markdown("### 📏 Your Input Values")
    input_col1, input_col2, input_col3, input_col4 = st.columns(4)
    input_col1.metric("Sepal Length", f"{sepal_length} cm")
    input_col2.metric("Sepal Width", f"{sepal_width} cm")
    input_col3.metric("Petal Length", f"{petal_length} cm")
    input_col4.metric("Petal Width", f"{petal_width} cm")
    
    # Predict button
    if st.button("🔮 Predict Species", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'SepalLengthCm': [sepal_length],
            'SepalWidthCm': [sepal_width],
            'PetalLengthCm': [petal_length],
            'PetalWidthCm': [petal_width]
        })
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_species = species_map[prediction_encoded]
        emoji = species_emoji[prediction_species]
        
        # Display prediction
        st.markdown("---")
        st.success("Prediction Complete!")
        
        # Show result in a big box
        st.markdown(f"""
        <div style='padding: 3rem; border-radius: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; text-align: center; margin: 2rem 0;'>
            <h1 style='font-size: 3rem; margin: 0;'>{emoji} {prediction_species} {emoji}</h1>
            <p style='font-size: 1.5rem; margin-top: 1rem;'>Predicted Species</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show prediction details
        st.markdown("### 📋 Prediction Details")
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        detail_col1.metric("Sepal Length", f"{sepal_length} cm", "Input")
        detail_col2.metric("Sepal Width", f"{sepal_width} cm", "Input")
        detail_col3.metric("Petal Length", f"{petal_length} cm", "Input")
        detail_col4.metric("Petal Width", f"{petal_width} cm", "Input")
        
        # Get decision scores
        try:
            decision_scores = model.decision_function(input_scaled)[0]
            with st.expander("🎯 View Model Confidence Scores"):
                st.write("**Decision Function Scores:**")
                for i, species in species_map.items():
                    score = decision_scores[i]
                    st.write(f"{species_emoji[species]} **{species}**: {score:.3f}")
        except:
            pass

with tab2:
    st.subheader("📁 Upload CSV for Batch Predictions")
    st.write("Upload a CSV file with columns: `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`")
    
    # Sample data
    sample_data = pd.DataFrame({
        'SepalLengthCm': [5.1, 6.2, 7.3, 4.9],
        'SepalWidthCm': [3.5, 2.9, 2.9, 3.1],
        'PetalLengthCm': [1.4, 4.3, 6.3, 1.5],
        'PetalWidthCm': [0.2, 1.3, 1.8, 0.2]
    })
    
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_data.to_csv(index=False),
        file_name="iris_sample.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("**Preview of uploaded data:**")
            st.dataframe(df_batch.head(10))
            
            if st.button("🎯 Make Batch Predictions", type="primary"):
                # Scale and predict
                df_scaled = scaler.transform(df_batch)
                predictions_encoded = model.predict(df_scaled)
                predictions_species = [species_map[pred] for pred in predictions_encoded]
                
                # Add predictions to dataframe
                df_batch['Predicted_Species'] = predictions_species
                df_batch['Emoji'] = [species_emoji[sp] for sp in predictions_species]
                
                st.success(f"✅ Predictions completed for {len(df_batch)} samples!")
                st.dataframe(df_batch)
                
                # Show counts
                st.markdown("### 📊 Prediction Summary")
                col1, col2, col3 = st.columns(3)
                
                setosa_count = predictions_species.count("Iris-setosa")
                versicolor_count = predictions_species.count("Iris-versicolor")
                virginica_count = predictions_species.count("Iris-virginica")
                
                col1.metric("🌺 Iris-setosa", setosa_count)
                col2.metric("🌸 Iris-versicolor", versicolor_count)
                col3.metric("🌼 Iris-virginica", virginica_count)
                
                # Download results
                csv = df_batch.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="iris_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | SVM Model with 100% Accuracy ✨")