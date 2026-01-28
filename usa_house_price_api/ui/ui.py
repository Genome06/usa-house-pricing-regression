import streamlit as st
import requests

st.set_page_config(page_title="USA House Price Predictor", layout="wide")

st.title("🏠 USA House Price Prediction")
st.markdown("Predict property values using advanced **XGBoost** regression model.")

# Input Sidebar
with st.sidebar:
    st.header("📋 Property Details")
    sqft_living = st.number_input("Living Area (sqft)", min_value=300, value=2000)
    bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, value=2.5, step=0.5)
    floors = st.number_input("Floors", min_value=1, value=2)
    sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, value=0)
    city = st.text_input("City", value="Seattle")
    statezip = st.text_input("Statezip", value="WA 98103")
    view = st.slider("View Quality (0-4)", 0, 4, 0)
    submit = st.button("💰 Predict Price")

if not submit:
    st.info("👈 Please enter property details in the sidebar to get a price estimation.")
    st.image("ui/image/feature_importance.png", caption="Feature Importance of the Model")

# Output Prediction
if submit:
    payload = {
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "floors": floors,
        "view": view,
        "sqft_basement": sqft_basement,
        "city": city,
        "statezip": statezip,
        "bedrooms": bedrooms
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        res_data = response.json()

        if res_data["status"] == "success":
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Predicted Price (USD)", f"${res_data['prediction_usd']:,.2f}")
            with col_res2:
                st.metric("Model Confidence (R²)", "75.47%")
            st.info(f"Model R² Score: {res_data['model_r2_score']}")
        else:
            st.error(f"Error: {res_data['message']}")
    except Exception as e:
        st.error(f"Could not connect to API. Is FastAPI running? (Error: {e})")