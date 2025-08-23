import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Model score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# --- Streamlit UI ---
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

st.title("🏠 House Price Prediction App")
st.markdown("This app predicts **Median House Prices in California** based on housing and demographic details. "
            "It is trained on the **California Housing Dataset** using Linear Regression.")

# Input fields (user-friendly)
st.subheader("📌 Enter Housing Details")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income (in $10,000)", 0.0, 15.0, 3.0, 0.1, help="Average income of households in the area")
    HouseAge = st.slider("Average House Age (years)", 1, 100, 20, help="Median age of houses in the area")
    AveRooms = st.slider("Average Number of Rooms", 1.0, 15.0, 5.0, 0.1, help="Average rooms per house")
    AveBedrms = st.slider("Average Number of Bedrooms", 0.5, 5.0, 1.0, 0.1, help="Average bedrooms per house")

with col2:
    Population = st.slider("Population of the Area", 100, 50000, 1000, 100, help="Total population in the area")
    AveOccup = st.slider("Average Occupancy per Household", 1.0, 10.0, 3.0, 0.1, help="Average household size")
    Latitude = st.slider("Latitude", 32.0, 42.0, 34.0, 0.1, help="Geographical latitude")
    Longitude = st.slider("Longitude", -124.0, -114.0, -118.0, 0.1, help="Geographical longitude")

# Prediction
if st.button("🔮 Predict House Price"):
    input_data = pd.DataFrame(
        [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
        columns=X.columns
    )
    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")
    st.success(f"💰 Estimated Median House Price: **${prediction*100000:,.2f}**")

    st.markdown(f"📈 **Model Accuracy (R² Score):** `{r2:.2f}`")
    st.caption("ℹ️ R² Score indicates how well the model explains the data (1.0 = perfect).")

    # Useful extra insights
    st.subheader("💡 Insights Based on Inputs")
    if MedInc > 6:
        st.info("🏦 High median income → Houses in this area are expected to be expensive.")
    elif MedInc < 2:
        st.warning("⚠️ Low median income → Houses are likely to be cheaper.")

    if Population > 20000:
        st.info("👥 High population density → May affect housing demand & prices.")
    else:
        st.info("🌿 Low population density → Quieter neighborhoods, possibly cheaper housing.")
