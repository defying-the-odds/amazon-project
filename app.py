import streamlit as st
import requests
from ml_module import get_recommendations

# Title
st.title("Product Recommendation System")

# User Input Fields
keywords = st.text_input("Enter keywords (e.g., 'running shoes')", "")
price = st.number_input("Maximum price (optional)", min_value=0.0, value=0.0, step=10.0)
stars = st.slider("Minimum star rating (optional)", min_value=0.0, max_value=5.0, value=0.0, step=0.5)

# Submit Button
if st.button("Get Recommendations"):
    # Prepare query parameters (only include non-default values)
    query_params = {}
    if keywords:
        query_params["keywords"] = keywords
    if price > 0:  # Only include if user sets a value greater than 0
        query_params["price"] = price
    if stars > 0:  # Only include if user sets a minimum
        query_params["stars"] = stars

    # Check if price is 0 before proceeding
    if price == 0:
        st.warning("You can't buy products for free! Please set a maximum price greater than 0.")
    elif query_params:  # Proceed only if there are valid parameters and price > 0
        try:
            # Send POST request to backend
            response = requests.post(
                "http://127.0.0.1:8000/recommend",
                json=query_params
            )
            if response.status_code == 200:
                data = response.json()
                if data["recommendations"]:
                    st.write("### Recommended Products:")
                    for product in data["recommendations"]:
                        st.markdown("----")
                        cols = st.columns([1, 3])  # Left for image, right for details

                        with cols[0]:
                            if product.get("imgURL"):
                                st.image(product["imgURL"], width=120)

                        with cols[1]:
                            title = product.get("title", "Unnamed Product")
                            title = product.get("title", "Unnamed Product")
                            product_url = f"https://www.amazon.com/s?k={title.replace(' ', '+')}"

                            if product_url and product_url.startswith("http"):
                                st.markdown(f'<a href="{product_url}" target="_blank"><strong>{title}</strong></a>', unsafe_allow_html=True)
                            else:
                                st.markdown(f"**{title}**")

                            st.write(f"${product['price']:.2f}")
                            if "rating" in product:
                                st.write(f"Rating: {product['rating']} ★ ({product.get('review_count', 0)} reviews)")
                            if "category" in product:
                                st.write(f"Category: {product['category']}")

                else:
                    st.write("No recommendations found. Try adjusting your query.")
            else:
                st.error(f"Error fetching recommendations: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter at least one preference (keywords, price, or stars).")