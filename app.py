from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Mantenimiento Predictivo Industrial", page_icon="🏭", layout="wide")

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_mantenimiento.pkl"
LABELS = {
    0: "Operación Normal",
    1: "Mantenimiento Preventivo",
    2: "Fallo Crítico",
}

MODELOS_EXCLUIDOS = ["whisper", "guard", "tts", "playai", "embed", "orpheus", "safeguard"]
MODELOS_PREFERIDOS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


def obtener_modelo_groq(client):
    modelos_activos = [
        m.id
        for m in client.models.list().data
        if not any(x in m.id.lower() for x in MODELOS_EXCLUIDOS)
    ]
    if not modelos_activos:
        raise RuntimeError("No se encontraron modelos de chat activos en Groq.")
    for preferido in MODELOS_PREFERIDOS:
        if preferido in modelos_activos:
            return preferido
    return next((m for m in modelos_activos if "llama" in m.lower()), modelos_activos[0])


def construir_prompt(estado, etiqueta, datos, probabilidad):
    base = f"""Datos de telemetría de maquinaria:
- Temperatura: {datos['temperatura_c']} °C
- Vibración: {datos['vibracion_mms']} mm/s
- Horas de operación: {datos['horas_operacion']} hrs
- Presión: {datos['presion_psi']} PSI
- Diagnóstico ML: {etiqueta} (estado {estado})
- Probabilidad estimada: {probabilidad:.1f}%

"""
    if estado == 0:
        return (
            base
            + "El equipo opera dentro de parámetros normales. Redacta un dictamen breve de monitoreo preventivo en Markdown, "
            + "confirmando estabilidad, indicando qué variables vigilar y una recomendación de seguimiento. Sé conciso."
        )
    urgencia = "preventiva programada" if estado == 1 else "urgente por riesgo inminente de avería"
    return (
        base
        + f"Genera una Orden de Trabajo Técnica en Markdown con severidad {urgencia}. "
        + "Incluye exactamente estas secciones: "
        + "## Orden de Trabajo\n"
        + "- **Folio:** (genera un folio breve)\n"
        + "- **Diagnóstico:** (resumen técnico basado en los sensores)\n"
        + "### 3 Acciones Técnicas\n"
        + "(lista numerada con 3 acciones concretas)\n"
        + "### Equipo de Seguridad (EPP)\n"
        + "(lista con el EPP requerido)\n"
        + "Adapta la urgencia al estado diagnosticado."
    )


st.title("🏭 Sistema Híbrido de Mantenimiento Predictivo")
st.write(
    "Monitoreo de telemetría industrial con **Machine Learning (Scikit-Learn)** "
    "y generación de órdenes de trabajo con **Groq API**."
)

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar 'modelo_mantenimiento.pkl': {e}")
    st.stop()

st.sidebar.header("Lecturas de Maquinaria")
temperatura = st.sidebar.slider("Temperatura (°C)", 40.0, 120.0, 62.5, 0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 15.0, 1.8, 0.1)
horas = st.sidebar.slider("Horas de operación (hrs)", 0, 6000, 450, 10)
presion = st.sidebar.slider("Presión (PSI)", 70.0, 180.0, 95.0, 0.5)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

if st.button("Evaluar Estado de Maquinaria", type="primary"):
    datos = {
        "temperatura_c": temperatura,
        "vibracion_mms": vibracion,
        "horas_operacion": horas,
        "presion_psi": presion,
    }
    entrada = pd.DataFrame([datos])

    prediccion = int(modelo.predict(entrada)[0])
    probabilidades = modelo.predict_proba(entrada)[0]
    clases = [int(c) for c in modelo.classes_]
    prob_pred = float(probabilidades[clases.index(prediccion)] * 100)
    etiqueta = LABELS.get(prediccion, f"Estado {prediccion}")

    st.subheader("Diagnóstico del Modelo Predictivo")
    mensaje = f"**{etiqueta}** (Probabilidad: {prob_pred:.1f}%)"
    if prediccion == 0:
        st.success(f"✅ {mensaje}")
    elif prediccion == 1:
        st.warning(f"⚠️ {mensaje}")
    else:
        st.error(f"🚨 {mensaje}")

    with st.expander("Probabilidades por clase"):
        probs_df = pd.DataFrame(
            {
                "Estado": [LABELS.get(int(c), str(c)) for c in clases],
                "Probabilidad (%)": [round(float(p) * 100, 2) for p in probabilidades],
            }
        )
        st.dataframe(probs_df, hide_index=True, width="stretch")

    st.subheader("Orden de Trabajo / Dictamen (Groq API)")
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = ""

    if not api_key or "TU_API_KEY" in api_key:
        st.warning(
            "Groq API no configurada. Se muestra el diagnóstico ML; configura `GROQ_API_KEY` en `.streamlit/secrets.toml` o en Streamlit Cloud."
        )
    else:
        try:
            client = Groq(api_key=api_key)
            with st.spinner("Generando dictamen técnico con Groq..."):
                modelo_groq = obtener_modelo_groq(client)
                prompt_usuario = construir_prompt(prediccion, etiqueta, datos, prob_pred)
                completion = client.chat.completions.create(
                    model=modelo_groq,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un Jefe de Mantenimiento Industrial experto en diagnóstico de maquinaria, "
                                "seguridad operativa y elaboración de órdenes de trabajo técnicas. "
                                "Responde en español, con formato Markdown claro y profesional."
                            ),
                        },
                        {"role": "user", "content": prompt_usuario},
                    ],
                    temperature=0.4,
                    max_tokens=900,
                )
            st.caption(f"Respuesta generada con el modelo: `{modelo_groq}`")
            st.markdown(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"Error al consultar Groq API: {e}")
            st.info("El diagnóstico de Machine Learning se mantiene disponible arriba.")