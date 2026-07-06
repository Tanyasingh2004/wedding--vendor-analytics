import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("final_wedding_data.csv")

# Label Encoders
le_city = LabelEncoder()
le_category_x = LabelEncoder()

le_city.fit(df['city'])
le_category_x.fit(df['category_x'])

# Load ML model
model = joblib.load("wedding_model.pkl")

# Title
st.title("Wedding Vendor Analytics Platform")

#graphs
st.subheader("Revenue by city")
city_revenue = df.groupby('city')['price_paid'].sum()
st.bar_chart(city_revenue)
st.subheader("Vendor Category Distribution")
fig, ax = plt.subplots()
df['category_x'].value_counts().plot(kind='pie',autopct='%1.1f%%',ax=ax)
st.pyplot(fig)
st.subheader("Monthly Wedding Revenue")
df['booking_date'] = pd.to_datetime(df['booking_date'])
monthly = df.groupby(df['booking_date'].dt.month)['price_paid'].sum()
st.line_chart(monthly)

# Prediction Section
st.subheader("Wedding Cost Prediction")

# City dropdown
city = st.selectbox(
    "Select City",
    le_city.classes_
)

# Category dropdown
category = st.selectbox(
    "Select Category",
    le_category_x.classes_
)

# Rating slider
rating = st.slider(
    "Vendor Rating",
    1.0,
    5.0,
    4.0
)

# Encode inputs
city_encoded = le_city.transform([city])[0]

category_encoded = le_category_x.transform([category])[0]

# Predict Button
if st.button("Predict Cost"):

    input_data = pd.DataFrame({
        'city': [city_encoded],
        'category_x': [category_encoded],
        'rating': [rating]
    })

    result = model.predict(input_data)

    st.success(
        f"Estimated Wedding Cost: ₹{result[0]:,.2f}"
    )