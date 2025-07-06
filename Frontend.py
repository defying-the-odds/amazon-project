import streamlit as st
import requests

# Configure page
st.set_page_config(page_title="Product Recommendations", layout="wide")

st.markdown("""
<style>
    body, .main {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #1e3d59;
        text-align: center;
    }

    .stTextInput label, .stNumberInput label, .stSlider label {
        font-weight: 600;
        color: #1e3d59;
        font-size: 16px;
    }

    .stTextInput > div > div > input,
    .stNumberInput > div > input {
        border: 2px solid #1e3d59 !important;
        border-radius: 6px;
        padding: 0.5rem;
        background-color: #ffffff;
    }

    .stSlider > div[data-baseweb="slider"] > div {
        background: #1e3d59;
    }

    .stButton button {
        background-color: #ff6b6b !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 18px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }

    .product-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #1e3d59;
        border-right: 5px solid #ff6b6b;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .recommendation-header {
        background-color: #1e3d59;
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 25px;
        text-align: center;
    }

    .filter-container {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        max-width: 650px;
        margin: 20px auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    hr {
        border: 1px solid #e2e8f0;
        margin: 25px 0;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("<h1 style='text-align: center; color: #1e3d59;'>Smart Product Recommendations</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #1e3d59;'>Find the perfect products based on your preferences</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# Input Form Container
st.markdown("<div class='filter-container'>", unsafe_allow_html=True)

keywords = st.text_input("🔍 Enter product keywords", placeholder="e.g., running shoes")
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

price = st.number_input("💵 Maximum price", min_value=0.0, value=100.0, step=5.0)
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

stars = st.slider("⭐ Minimum star rating", min_value=0.0, max_value=5.0, value=0.0, step=0.5)
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

search_button = st.button("🔎 Get Recommendations", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# API call logic
def get_recommendations(query_params):
    try:
        response = requests.post("http://127.0.0.1:8000/recommend", json=query_params)
        return response.json()
    except Exception as e:
        st.warning(f"Could not connect to the real API: {e}. Using mock data instead.")
        mock_data = {
            "recommendations": [
                {"title": "Sample Premium Edition", "price": 79.99, "rating": 4.5, "category": "Electronics"},
                {"title": "Sample Standard Model", "price": 49.99, "rating": 4.2, "category": "Electronics"},
                {"title": "Sample Budget Version", "price": 29.99, "rating": 3.8, "category": "Electronics"}
            ]
        }
        return mock_data

# Handle recommendation display
if search_button and keywords:
    query_params = {"keywords": keywords}
    if price > 0: query_params["price"] = price
    if stars > 0: query_params["stars"] = stars

    with st.spinner("Finding the best products for you..."):
        result = get_recommendations(query_params)

    if result.get("recommendations"):
        st.markdown("<div class='recommendation-header'><h2 style='text-align: center;'>Recommended Products</h2></div>", unsafe_allow_html=True)

        for product in result["recommendations"]:
            st.markdown("<div class='product-card'>", unsafe_allow_html=True)

            title = product.get("title", "Unnamed Product")
            product_url = f"https://www.amazon.com/s?k={title.replace(' ', '+')}"

            st.markdown(f"<h3><a href='{product_url}' target='_blank' style='text-decoration:none; color:#1e3d59;'>{title}</a></h3>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Price:</strong> ${product.get('price', 0.0):.2f}</p>", unsafe_allow_html=True)
            if "rating" in product:
                st.markdown(f"<p><strong>Rating:</strong> {'⭐' * int(float(product['rating']))} ({float(product['rating']):.1f})</p>", unsafe_allow_html=True)
            if "category" in product:
                st.markdown(f"<p><strong>Category:</strong> {product['category']}</p>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No recommendations found. Try different criteria.")
else:
    st.info("Enter product keywords and click 'Get Recommendations'.")

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 50px; padding: 20px; color: #666;'>
    <p>© 2025 Product Recommendation System | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
