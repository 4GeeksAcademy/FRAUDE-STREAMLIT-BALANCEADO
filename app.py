import joblib
import streamlit as st
import os
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Configuración de la aplicación
st.set_page_config(page_title="Fraude Financiero", page_icon="💰", layout="wide")

# Ruta del modelo
RUTA_MODELO = "modelo_RandomForest_optimizado.pkl.gz"

def cargar_modelo_comprimido(ruta):
    """Carga el modelo comprimido con gzip."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"El archivo {ruta} no existe. Verifica que está en la carpeta correcta.")
    with gzip.open(ruta, "rb") as f:
        modelo = joblib.load(f)
    return modelo

# Cargar el modelo
try:
    model = cargar_modelo_comprimido(RUTA_MODELO)
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Verificar si el modelo es un RandomForestClassifier
if not isinstance(model, RandomForestClassifier):
    st.error("El archivo cargado no es un modelo RandomForest.")
    st.stop()

# Diccionario de clases
class_dict = {"0": "No Fraude", "1": "Fraude"}

# Barra lateral
menu = st.sidebar.radio("📌 Menú de Navegación", ["Predicción de Fraude", "Reseña sobre Fraudes Financieros"])

if menu == "Predicción de Fraude":
    st.title("🔍 Predicción de Fraude en Transacciones Bancarias")

    with st.form("formulario_prediccion"):  
        st.subheader("📊 Ingrese los Datos de la Transacción")
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Ingresos", min_value=0.0, max_value=10000000.0, step=1000.0, value=1000.0)
            name_email_similarity = st.slider("Similitud entre Nombre y Email", 0.0, 1.0, 0.5, step=0.01)
            prev_address_months_count = st.number_input("Meses en Dirección Anterior", min_value=0, max_value=240, value=12)
            current_address_months_count = st.number_input("Meses en Dirección Actual", min_value=0, max_value=240, value=12)
            customer_age = st.number_input("Edad del Cliente", min_value=18, max_value=100, value=30)
            velocity_6h = st.number_input("Velocidad Transacción 6h", min_value=0.0, max_value=1000.0, value=10.0)
            velocity_24h = st.number_input("Velocidad Transacción 24h", min_value=0.0, max_value=1000.0, value=20.0)
            has_other_cards = st.radio("¿Tiene Otras Tarjetas?", ["No", "Sí"], index=0)
            foreign_request = st.radio("¿Solicitud Extranjera?", ["No", "Sí"], index=0)
        
        with col2:
            proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", min_value=0.0, max_value=1000000.0, value=5000.0)
            bank_branch_count_8w = st.number_input("Sucursales Bancarias 8 Semanas", min_value=0, max_value=20, value=5)
            credit_risk_score = st.number_input("Puntuación de Riesgo Crediticio", min_value=0, max_value=1000, value=300)
            keep_alive_session = st.number_input("Duración Sesión Activa (min)", min_value=0.0, max_value=1440.0, value=60.0)
            month = st.slider("Mes de la Transacción", min_value=1, max_value=12, value=1)
            email_is_free = st.radio("¿Email Gratuito?", ["No", "Sí"], index=0)
            phone_home_valid = st.radio("¿Teléfono Casa Válido?", ["No", "Sí"], index=0)
            phone_mobile_valid = st.radio("¿Teléfono Móvil Válido?", ["No", "Sí"], index=0)

        submit_button = st.form_submit_button("🚀 Predecir")  

    if submit_button:  
        data_df = pd.DataFrame([[
            income, name_email_similarity, prev_address_months_count, current_address_months_count, customer_age,
            velocity_6h, velocity_24h, bank_branch_count_8w, credit_risk_score, email_is_free == "Sí",
            phone_home_valid == "Sí", phone_mobile_valid == "Sí", has_other_cards == "Sí", proposed_credit_limit,
            foreign_request == "Sí", keep_alive_session, month
        ]], columns=[
            'income', 'name_email_similarity', 'prev_address_months_count',
            'current_address_months_count', 'customer_age', 'velocity_6h',
            'velocity_24h', 'bank_branch_count_8w', 'credit_risk_score',
            'email_is_free', 'phone_home_valid', 'phone_mobile_valid',
            'has_other_cards', 'proposed_credit_limit', 'foreign_request',
            'keep_alive_session', 'month'
        ])
        
        try:
            prediction = str(model.predict(data_df)[0])
            pred_class = class_dict[prediction]
            st.success(f"🔮 **Predicción:** {pred_class}")
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")

elif menu == "Reseña sobre Fraudes Financieros":
    st.title("📖 Reseña sobre Fraudes Financieros")
    st.markdown("""  
    Los fraudes financieros buscan engañar a individuos o empresas para obtener dinero de forma ilícita.
    
    ### 📌 Cómo se Combate el Fraude Financiero:
    - **Machine Learning**: Identifica patrones sospechosos en tiempo real.
    - **Autenticación de Múltiples Factores (MFA)**.
    - **Educación Financiera**.
    
    ### 📈 Métodos de Machine Learning:
    - **Supervisados:** Random Forest, XGBoost.
    - **No Supervisados:** Clustering, Isolation Forest.
    """)

    st.subheader("📊 Distribución de Fraudes por Región")
    regiones = ["Norteamérica", "Europa", "Latinoamérica", "Asia"]
    fraudes = [3000, 2500, 1800, 2200]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=regiones, y=fraudes, palette="viridis", ax=ax)
    ax.set_xlabel("Región")
    ax.set_ylabel("Casos de Fraude")
    st.pyplot(fig)