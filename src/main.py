import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# Generate sample dataset
def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({"size": size, "price": price})


# Train the model
def train_model():
    df = generate_house_data(n_samples=100)
    X = df[["size"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def main():
    st.title("Simple Linear Regression House Price Predictor")
    st.write("Enter a house size to estimate its price:")

    model = train_model()

    # Streamlit input
    size = st.number_input(
        "House size (sq ft)", min_value=500, max_value=2000, value=1400
    )

    if st.button("Predict price"):
        predicted_price = model.predict([[size]])
        st.success(f"Estimated price: ${predicted_price[0]:,.2f}")

        # Plot
        df = generate_house_data()
        fig = px.scatter(df, x="size", y="price", title="Size vs Price")
        fig.add_scatter(
            x=[size],
            y=[predicted_price[0]],
            mode="markers",
            marker=dict(size=15, color="red"),
            name="Prediction",
        )
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
