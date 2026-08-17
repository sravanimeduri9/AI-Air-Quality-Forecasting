import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Air Quality Forecasting",
    page_icon="🌍",
    layout="wide"
)

# Load trained model
model = joblib.load("model/aqi_model.pkl")
features = joblib.load("model/features.pkl")

# Load city data
data = pd.read_csv("Data/cleaned_city_day.csv")

cities = sorted(data["City"].dropna().unique())

selected_city = st.selectbox(
    "🏙️ Select City",
    cities
)

# Get latest data for selected city
city_data = data[data["City"] == selected_city].sort_values("Date")

latest_data = city_data.iloc[-1]

st.success(f"📍 Selected City: {selected_city}")

st.write("Latest available air-quality readings:")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("PM2.5", f"{latest_data['PM2.5']:.2f}")
    st.metric("PM10", f"{latest_data['PM10']:.2f}")
    st.metric("NO2", f"{latest_data['NO2']:.2f}")
    st.metric("CO", f"{latest_data['CO']:.2f}")

with col2:
    st.metric("NO", f"{latest_data['NO']:.2f}")
    st.metric("NOx", f"{latest_data['NOx']:.2f}")
    st.metric("NH3", f"{latest_data['NH3']:.2f}")
    st.metric("SO2", f"{latest_data['SO2']:.2f}")

with col3:
    st.metric("O3", f"{latest_data['O3']:.2f}")
    st.metric("Benzene", f"{latest_data['Benzene']:.2f}")
    st.metric("Toluene", f"{latest_data['Toluene']:.2f}")
    st.metric("Xylene", f"{latest_data['Xylene']:.2f}")

st.write("Selected City:", selected_city)

# Health advisory function
def get_advisory(aqi):
    if aqi <= 50:
        return "Good", "Air quality is good. Normal outdoor activities are generally okay."
    elif aqi <= 100:
        return "Satisfactory", "Air quality is acceptable. Sensitive people should be a little careful."
    elif aqi <= 200:
        return "Moderate", "Sensitive people should consider reducing prolonged outdoor activity."
    elif aqi <= 300:
        return "Poor", "Reduce prolonged outdoor exposure and follow appropriate precautions."
    elif aqi <= 400:
        return "Very Poor", "Avoid unnecessary outdoor activities and follow local guidance."
    else:
        return "Severe", "Avoid outdoor exposure as much as possible and follow local health guidance."

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🌍 AI Air Quality Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">🤖 Personalized Health Advisory System</div>',
    unsafe_allow_html=True
)

st.write("")
# Title
st.info(
    "This AI-based system uses air-quality parameters and a "
    "Random Forest Machine Learning model to estimate AQI "
    "and provide a general health advisory."
)


st.divider()

# Input section
st.subheader("🌫️ Air Quality Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
    pm10 = st.number_input("PM10", min_value=0.0, value=80.0)
    no = st.number_input("NO", min_value=0.0, value=20.0)
    no2 = st.number_input("NO2", min_value=0.0, value=30.0)

with col2:
    nox = st.number_input("NOx", min_value=0.0, value=40.0)
    nh3 = st.number_input("NH3", min_value=0.0, value=20.0)
    co = st.number_input("CO", min_value=0.0, value=1.0)
    so2 = st.number_input("SO2", min_value=0.0, value=20.0)

with col3:
    o3 = st.number_input("O3", min_value=0.0, value=30.0)
    benzene = st.number_input("Benzene", min_value=0.0, value=2.0)
    toluene = st.number_input("Toluene", min_value=0.0, value=5.0)
    xylene = st.number_input("Xylene", min_value=0.0, value=2.0)


# Prediction button
if st.button("🔮 Predict AQI", 
use_container_width=True):

     # Use selected city's latest air-quality readings
    input_data = pd.DataFrame(
        [[
            latest_data["PM2.5"],
            latest_data["PM10"],
            latest_data["NO"],
            latest_data["NO2"],
            latest_data["NOx"],
            latest_data["NH3"],
            latest_data["CO"],
            latest_data["SO2"],
            latest_data["O3"],
            latest_data["Benzene"],
            latest_data["Toluene"],
            latest_data["Xylene"]
        ]],
        columns=features
    )

    # Predict AQI
    predicted_aqi = model.predict(input_data)[0]

    # Get category and advisory
    category, advice = get_advisory(predicted_aqi)

    st.divider()

    st.subheader("📊 AQI Prediction Result")

    result1, result2 = st.columns(2)

    with result1:
        st.metric("Predicted AQI", f"{predicted_aqi:.2f}")

    with result2:
        st.metric("AQI Category", category)

    st.subheader("💡 Personalized Health Advisory")

    st.info(advice)

    st.caption(
        "This is general informational guidance and is not a medical diagnosis."
    )

    # Project Information

    st.divider()

    st.subheader("📌 About This Project")

    st.write("""
    This project uses Artificial Intelligence and Machine Learning
    to forecast Air Quality Index (AQI) from air-quality parameters.

    The system uses a Random Forest Regression model to predict AQI
    and classifies the predicted value into different air-quality
    categories.

    Based on the AQI category, the system provides a general
    personalized health advisory to the user.
    """)

    st.subheader("🔄 System Workflow")

    st.write("""
    1. User selects a city.
    2. The system retrieves the latest available air-quality data.
    3. Air-quality parameters are given to the trained ML model.
    4. Random Forest predicts the AQI.
    5. The predicted AQI is classified into an AQI category.
    6. A general health advisory is displayed.
    """)

    # AQI Status
    if predicted_aqi <= 50:
        st.success("🟢 Good Air Quality")

    elif predicted_aqi <= 100:
        st.info("🟡 Satisfactory Air Quality")

    elif predicted_aqi <= 200:
        st.warning("🟠 Moderate Air Quality")

    elif predicted_aqi <= 300:
        st.warning("🔴 Poor Air Quality")

    elif predicted_aqi <= 400:
        st.error("🟣 Very Poor Air Quality")

    else:
        st.error("⚠️ Severe Air Quality")