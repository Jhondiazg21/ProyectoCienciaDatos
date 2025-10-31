import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATA_DIR = Path.cwd()  # Ajusta la ruta según sea necesario

# Cargar el modelo previamente guardado
loaded_model = joblib.load(DATA_DIR / "models" / "trained" / "heart_failure_clinical_records-logistic_regression-v1.joblib")
print("Modelo cargado exitosamente.")

st.title('Predictor de Riesgo de Insuficiencia Cardíaca')

st.write("""
### Por favor ingrese los datos del paciente:
""")

# Crear columnas para mejor organización visual
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Edad', min_value=40, max_value=95, value=60)
    anaemia = st.checkbox('Anemia')
    creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase (mcg/L)', min_value=23, max_value=7861, value=500)
    diabetes = st.checkbox('Diabetes')
    ejection_fraction = st.number_input('Fracción de Eyección (%)', min_value=14, max_value=80, value=38)
    high_blood_pressure = st.checkbox('Hipertensión')

with col2:
    platelets = st.number_input('Plaquetas (kiloplaquetas/mL)', min_value=100000.0, max_value=850000.0, value=250000.0)
    serum_creatinine = st.number_input('Creatinina Sérica (mg/dL)', min_value=0.5, max_value=9.4, value=1.1)
    serum_sodium = st.number_input('Sodio Sérico (mEq/L)', min_value=113, max_value=148, value=136)
    sex = st.radio('Sexo', ['Femenino', 'Masculino'])
    smoking = st.checkbox('Fumador')
    time = st.number_input('Tiempo de seguimiento (días)', min_value=4, max_value=285, value=130)

if st.button('Predecir Riesgo'):
    # Preparar los datos en el formato correcto
    data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex == 'Masculino',  # Convertir a booleano
        'smoking': smoking,
        'time': time
    }
    
    # Convertir a DataFrame
    input_df = pd.DataFrame([data])
    
    # Asegurar tipos de datos correctos
    numeric_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                   'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    boolean_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    
    # Convertir columnas numéricas a float64
    for col in numeric_cols:
        input_df[col] = input_df[col].astype('float64')
    
    # Convertir columnas booleanas a int64
    for col in boolean_cols:
        input_df[col] = input_df[col].astype('int64')

    # Hacer la predicción
    prediction = loaded_model.predict(input_df)
    probability = loaded_model.predict_proba(input_df)
    
    # Mostrar resultados
    st.subheader('Resultados:')
    if prediction[0]:
        st.error('⚠️ Alto Riesgo de Evento Fatal')
        st.write(f'Probabilidad de evento fatal: {probability[0][1]:.2%}')
    else:
        st.success('✅ Bajo Riesgo de Evento Fatal')
        st.write(f'Probabilidad de evento fatal: {probability[0][1]:.2%}')

    # Mostrar recomendaciones
    st.subheader('Recomendaciones:')
    if prediction[0]:
        st.write("""
        - Consulte inmediatamente con su médico
        - Mantenga un seguimiento estricto de sus signos vitales
        - Siga rigurosamente su tratamiento médico
        """)
    else:
        st.write("""
        - Continúe con sus chequeos médicos regulares
        - Mantenga un estilo de vida saludable
        - Monitoree cualquier cambio en su condición
        """)