import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import sqlite3

class AmazonProductRecommender:
    def __init__(self):
        # Load data from SQLite
        conn = sqlite3.connect("products.db")
        self.product_data = pd.read_sql_query("SELECT * FROM products", conn)
        conn.close()

        # Map database columns to ML expected columns
        self.product_data = self.product_data.rename(columns={
            "stars": "rating",
            "reviews": "review_count",
            "category_id": "category",  # Assuming category_id acts as category
            "boughtInLastMonth": "sales_rank"  # Proxy for sales_rank
        })

        # Invert sales_rank (higher boughtInLastMonth = better)
        if "sales_rank" in self.product_data.columns:
            self.product_data["sales_rank"] = 1000000 / (self.product_data["sales_rank"] + 1)

        self.cosine_model = None
        self.cluster_model = None
        self.build_cosine_model()
        self.build_cluster_model()

    def preprocess_text(self, text):
        if isinstance(text, str):
            return re.sub(r'[^\w\s]', '', text.lower())
        return ""

    def build_cosine_model(self):
        text_features = []
        for _, row in self.product_data.iterrows():
            feature_text = f"{row.get('title', '')} {row.get('category', '')}"
            if 'sales_rank' in self.product_data.columns:
                sales_rank = row.get('sales_rank', 0)
                if sales_rank > 0:
                    popularity_level = min(10, max(1, int(1000000 / (sales_rank + 1000))))
                    feature_text += f" {'bestseller' * popularity_level}"
            if 'rating' in self.product_data.columns and 'review_count' in self.product_data.columns:
                rating = row.get('rating', 0)
                review_count = row.get('review_count', 0)
                if rating > 4.0 and review_count > 50:
                    feature_text += " highly rated well reviewed popular recommended"
            text_features.append(feature_text)

        self.product_data['combined_features'] = text_features
        self.product_data['combined_features'] = self.product_data['combined_features'].apply(self.preprocess_text)

        self.tfidf = TfidfVectorizer(stop_words='english')
        self.product_vectors = self.tfidf.fit_transform(self.product_data['combined_features'])
        print(f"Cosine similarity model built with {self.product_vectors.shape[0]} products")
        self.cosine_model = True

    def build_cluster_model(self, n_clusters=15):
        numerical_features = ['price']
        if 'rating' in self.product_data.columns:
            numerical_features.append('rating')
        if 'review_count' in self.product_data.columns:
            numerical_features.append('review_count')
        if 'sales_rank' in self.product_data.columns:
            self.product_data['sales_score'] = 1 / (self.product_data['sales_rank'] + 1)
            numerical_features.append('sales_score')

        features = self.product_data[numerical_features].copy()
        if 'category' in self.product_data.columns:
            top_categories = self.product_data['category'].value_counts().head(20).index
            self.product_data['top_category'] = self.product_data['category'].apply(
                lambda x: x if x in top_categories else 'Other'
            )
            category_dummies = pd.get_dummies(self.product_data['top_category'], prefix='category')
            features = pd.concat([features, category_dummies], axis=1)

        features = features.fillna(0)
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.product_data['cluster'] = self.kmeans.fit_predict(scaled_features)
        self.feature_columns = features.columns
        print(f"Cluster model built with {n_clusters} clusters")
        self.cluster_model = True

    def get_recommendations_cosine(self, query, top_n=10, min_rating=None, sort_price=None):
        if not self.cosine_model:
            self.build_cosine_model()

        enhanced_query = query
        if 'bestseller' in query.lower() or 'popular' in query.lower():
            enhanced_query += " bestseller popular top selling"
        if 'highly rated' in query.lower() or 'top rated' in query.lower():
            enhanced_query += " highly rated recommended"

        query_vector = self.tfidf.transform([self.preprocess_text(enhanced_query)])
        similarities = cosine_similarity(query_vector, self.product_vectors).flatten()

        max_price = float('inf')
        min_price = 0
        if 'under' in query.lower():
            match = re.search(r'under\s*\$?(\d+)', query.lower())
            if match:
                max_price = float(match.group(1))
        elif 'over' in query.lower():
            match = re.search(r'over\s*\$?(\d+)', query.lower())
            if match:
                min_price = float(match.group(1))

        filtered_indices = [
            i for i in range(len(similarities))
            if min_price <= self.product_data.iloc[i]['price'] <= max_price
        ]

        if 'bestseller' in query.lower() or 'popular' in query.lower():
            if 'sales_rank' in self.product_data.columns:
                sales_threshold = self.product_data['sales_rank'].quantile(0.2)
                filtered_indices = [
                    i for i in filtered_indices
                    if self.product_data.iloc[i]['sales_rank'] <= sales_threshold
                ]

        if 'highly rated' in query.lower() or 'top rated' in query.lower():
            if 'rating' in self.product_data.columns:
                filtered_indices = [
                    i for i in filtered_indices
                    if self.product_data.iloc[i]['rating'] >= 4.0
                ]
        if min_rating is not None:
            filtered_indices = [
                i for i in filtered_indices
                if self.product_data.iloc[i]['rating'] >= min_rating
    ]

        if filtered_indices:
            filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
            sorted_indices = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in sorted_indices[:top_n]]
            recommendations = self.product_data.iloc[top_indices]
            if sort_price == "Low to High":
                recommendations = recommendations.sort_values(by="price", ascending=True)
            elif sort_price == "High to Low":
                recommendations = recommendations.sort_values(by="price", ascending=False)

            result_fields = ['asin', 'title', 'category', 'price', 'rating', 'review_count', 'sales_rank', 'imgUrl', 'productURL']
            return recommendations[result_fields].to_dict('records')
        return []

    def get_recommendations_cluster(self, query, top_n=10, min_rating=None, sort_price=None):
        if not self.cluster_model:
            self.build_cluster_model()

        category = None
        for cat in self.product_data['category'].unique():
            if isinstance(cat, str) and cat.lower() in query.lower():
                category = cat
                break

        max_price = float('inf')
        if 'under' in query.lower():
            match = re.search(r'under\s*\$?(\d+)', query.lower())
            if match:
                max_price = float(match.group(1))

        test_features = pd.DataFrame(columns=self.feature_columns)
        test_features.loc[0, 'price'] = max_price / 2
        if 'sales_score' in self.feature_columns and ('bestseller' in query.lower() or 'popular' in query.lower()):
            test_features.loc[0, 'sales_score'] = 0.9
        if 'rating' in self.feature_columns and ('highly rated' in query.lower() or 'top rated' in query.lower()):
            test_features.loc[0, 'rating'] = 4.5
        if category:
            category_col = f'category_{category}'
            if category_col in self.feature_columns:
                test_features.loc[0, category_col] = 1

        test_features = test_features.fillna(0)
        for col in self.feature_columns:
            if col not in test_features.columns:
                test_features[col] = 0

        scaled_test_features = self.scaler.transform(test_features[self.feature_columns])
        cluster = self.kmeans.predict(scaled_test_features)[0]

        cluster_products = self.product_data[self.product_data['cluster'] == cluster].copy()

        if min_rating is not None:
            cluster_products = cluster_products[cluster_products['rating'] >= min_rating]


        if min_rating is not None:
            cluster_products = cluster_products[cluster_products['rating'] >= min_rating]

        if max_price < float('inf'):
            cluster_products = cluster_products[cluster_products['price'] <= max_price]
        if category and 'category' in self.product_data.columns:
            category_products = cluster_products[cluster_products['category'] == category]
            if not category_products.empty:
                cluster_products = category_products

        if 'bestseller' in query.lower() or 'popular' in query.lower():
            if 'sales_rank' in cluster_products.columns:
                cluster_products = cluster_products.sort_values('sales_rank')
        elif 'highly rated' in query.lower() or 'top rated' in query.lower():
            if 'rating' in cluster_products.columns:
                cluster_products = cluster_products.sort_values('rating', ascending=False)
        elif 'sales_rank' in cluster_products.columns:
            cluster_products = cluster_products.sort_values('sales_rank')

        if sort_price == "Low to High":
            cluster_products = cluster_products.sort_values(by="price", ascending=True)
        elif sort_price == "High to Low":
            cluster_products = cluster_products.sort_values(by="price", ascending=False)


        recommendations = cluster_products.head(top_n)
        result_fields = ['asin', 'title', 'category', 'price', 'rating', 'review_count', 'sales_rank', 'imgUrl', 'productURL']
        return recommendations[result_fields].to_dict('records')

    def get_recommendations(self, user_input, method='hybrid', top_n=10, min_rating=None, sort_price=None):

        if method == 'cosine':
            return self.get_recommendations_cosine(user_input, top_n, min_rating, sort_price)
        elif method == 'cluster':
            return self.get_recommendations_cluster(user_input, top_n, min_rating, sort_price)
        else:
            cosine_recs = self.get_recommendations_cosine(user_input, top_n, min_rating, sort_price)
            cluster_recs = self.get_recommendations_cluster(user_input, top_n, min_rating, sort_price)
            all_recs = cosine_recs + cluster_recs
            unique_recs = []
            seen_ids = set()
            for rec in all_recs:
                if rec['asin'] not in seen_ids:
                    unique_recs.append(rec)
                    seen_ids.add(rec['asin'])
                    if len(unique_recs) >= top_n:
                        break
            return unique_recs

# Singleton instance
recommender = AmazonProductRecommender()

def get_recommendations(query_params):
    """
    Wrapper for API integration
    Args:
        query_params (dict): From API (e.g., {"keywords": "running shoes", "price": 100.0})
    Returns:
        list: List of recommendation dicts
    """
    # Convert query_params dict to a string query
    query_str = ""
    if "keywords" in query_params:
        query_str += query_params["keywords"]
    if "price" in query_params:
        query_str += f" under ${query_params['price']}"
    if "stars" in query_params:
        query_str += f" highly rated" if query_params["stars"] >= 4.0 else ""
    if "boughtInLastMonth" in query_params and query_params["boughtInLastMonth"] > 100:
        query_str += " bestselling"

    return recommender.get_recommendations(
    query_str.strip(), 
    method='hybrid', 
    top_n=10, 
    min_rating=query_params.get("stars", None),
    sort_price=query_params.get("sort_price", None)
)
