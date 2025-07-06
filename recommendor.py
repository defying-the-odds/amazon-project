import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import random
import time
from collections import defaultdict
import whisper
import os
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Load Whisper model globally (use 'base' for speed in a hackathon)
model = whisper.load_model("base")

def convert_audio(input_path: str) -> str:
    """
    Converts audio to WAV (16-bit PCM, mono, 16kHz) for Whisper compatibility.
    Returns path to converted file.
    """
    try:
        output_path = "converted_audio.wav"
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)  # Whisper expects 16kHz mono
        audio.export(output_path, format="wav", codec="pcm_s16le")
        return output_path
    except CouldntDecodeError:
        raise Exception(f"Failed to decode audio file: {input_path}")
    except Exception as e:
        raise Exception(f"Audio conversion error: {str(e)}")

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes audio file using Whisper and returns the text.
    """
    try:
        print("Transcribing audio...")
        converted_path = convert_audio(file_path)
        result = model.transcribe(converted_path)
        transcript = result["text"].strip()
        
        print(f"Transcription complete: {transcript}")
        # Optionally save transcript (remove if not needed for hackathon)
        with open("transcript.txt", "w") as f:
            f.write(transcript)
        
        # Clean up temporary file
        if os.path.exists(converted_path):
            os.remove(converted_path)
        
        return transcript
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")

# QueryGenerator class remains unchanged
class QueryGenerator:
    def __init__(self, product_data):
        self.product_data = product_data
        self.categories = self.extract_categories()
        self.price_points = self.generate_price_points()
        self.query_templates = [
            "{category}",
            "{category} under ${price}",
            "best {category}",
            "bestselling {category}",
            "popular {category}",
            "top rated {category}",
            "highly rated {category}",
            "{category} between ${price_min} and ${price_max}",
            "affordable {category}",
            "premium {category}",
            "{category} with good reviews",
            "{adjective} {category}",
            "{brand} {category}",
            "{color} {category}",
            "{category} for {usage}",
            "{category} similar to {asin}",
            "alternatives to {brand} {category}",
            "{category} under ${price} with good reviews",
            "bestselling {category} under ${price}",
            "highly rated {category} under ${price}"
        ]
        self.adjectives = [
            "comfortable", "durable", "lightweight", "waterproof", "wireless", 
            "portable", "smart", "budget", "professional", "compact", "foldable",
            "noise-cancelling", "ergonomic", "adjustable", "fast", "powerful"
        ]
        self.colors = [
            "black", "white", "red", "blue", "green", "gray", "silver", "gold",
            "pink", "purple", "brown", "orange", "yellow"
        ]
        self.usages = [
            "home", "office", "travel", "gaming", "sports", "outdoors", "kids",
            "beginners", "professionals", "everyday use", "streaming", "work",
            "school", "exercise", "hiking", "camping", "cooking", "cleaning"
        ]
        self.brands = self.extract_brands()
        self.product_mapping = self.build_product_mapping()
        
    def extract_categories(self):
        if 'category' in self.product_data.columns:
            all_categories = self.product_data['category'].dropna().unique().tolist()
            categories = [cat for cat in all_categories if isinstance(cat, str) and len(cat.split()) < 4]
            return categories if categories else ["Electronics", "Books", "Home & Kitchen", "Clothing", "Sports"]
        return ["Electronics", "Books", "Home & Kitchen", "Clothing", "Sports"]
    
    def extract_brands(self):
        common_brands = ["Amazon", "Apple", "Samsung", "Sony", "LG", "Bose", "Nike", "Adidas",
                         "Logitech", "Microsoft", "Dell", "HP", "Anker", "JBL", "Canon", "Nikon"]
        if 'title' in self.product_data.columns:
            title_words = []
            for title in self.product_data['title'].dropna():
                if isinstance(title, str):
                    words = title.split()
                    if words and len(words[0]) > 2:
                        title_words.append(words[0])
            word_counts = defaultdict(int)
            for word in title_words:
                word_counts[word] += 1
            potential_brands = [word for word, count in word_counts.items() 
                              if count > 5 and word.isalpha()]
            return list(set(common_brands + potential_brands[:20]))
        return common_brands
    
    def generate_price_points(self):
        if 'price' in self.product_data.columns:
            prices = self.product_data['price'].dropna()
            if len(prices) > 0:
                min_price = max(1, int(prices.min()))
                max_price = min(1000, int(prices.max()))
                price_points = [
                    *range(min_price, min(50, max_price), 5),
                    *range(50, min(200, max_price), 25),
                    *range(200, min(max_price, 1000), 100)
                ]
                return sorted(list(set(price_points)))
        return [10, 20, 25, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    
    def build_product_mapping(self):
        mapping = defaultdict(list)
        if 'category' in self.product_data.columns and 'asin' in self.product_data.columns:
            for _, row in self.product_data.iterrows():
                category = row.get('category')
                asin = row.get('asin')
                if category and asin:
                    mapping[category].append(asin)
        return mapping
    
    def generate_query(self):
        template = random.choice(self.query_templates)
        if "{category}" in template:
            category = random.choice(self.categories)
            template = template.replace("{category}", category)
        if "{price}" in template:
            price = random.choice(self.price_points)
            template = template.replace("{price}", str(price))
        if "{price_min}" in template and "{price_max}" in template:
            prices = sorted(random.sample(self.price_points, 2))
            template = template.replace("{price_min}", str(prices[0]))
            template = template.replace("{price_max}", str(prices[1]))
        if "{adjective}" in template:
            adjective = random.choice(self.adjectives)
            template = template.replace("{adjective}", adjective)
        if "{color}" in template:
            color = random.choice(self.colors)
            template = template.replace("{color}", color)
        if "{usage}" in template:
            usage = random.choice(self.usages)
            template = template.replace("{usage}", usage)
        if "{brand}" in template:
            brand = random.choice(self.brands)
            template = template.replace("{brand}", brand)
        if "{asin}" in template:
            category = next((c for c in self.categories if c in template), random.choice(self.categories))
            asins = self.product_mapping.get(category, [])
            if asins:
                asin = random.choice(asins)
            else:
                asin = f"B{random.randint(10000000, 99999999)}"
            template = template.replace("{asin}", asin)
        return template
    
    def generate_queries(self, n=100):
        queries = set()
        while len(queries) < n:
            queries.add(self.generate_query())
        return list(queries)

class AmazonProductRecommender:
    def __init__(self, amazon_data):
        self.product_data = amazon_data
        self.cosine_model = None
        self.cluster_model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.query_generator = QueryGenerator(amazon_data)

    def build_cosine_model(self):
        print("Building cosine similarity model...")
        if 'title' not in self.product_data.columns:
            raise ValueError("Product data must contain a 'title' column")

        # Combine title and description
        if 'description' in self.product_data.columns:
            self.product_data['text'] = (
                self.product_data['title'].fillna('') + " " +
                self.product_data['description'].fillna('')
            )
        else:
            self.product_data['text'] = self.product_data['title'].fillna('')

        # Create and store TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.product_data['text'])
        self.cosine_model = cosine_similarity(self.tfidf_matrix)

    def build_cluster_model(self):
        print("Building cluster model...")
        if self.tfidf_matrix is None:
            self.build_cosine_model()
        
        # Standardize features
        scaler = StandardScaler(with_mean=False)
        scaled_features = scaler.fit_transform(self.tfidf_matrix)
        
        # Fit K-means
        self.cluster_model = KMeans(n_clusters=10, random_state=42)
        self.cluster_model.fit(scaled_features)
        self.product_data['cluster'] = self.cluster_model.labels_

    def get_recommendations(self, query, method='cosine', n=5):
        """Get product recommendations for a given query"""
        if method == 'cosine' and self.cosine_model is None:
            self.build_cosine_model()
        if method == 'cluster' and self.cluster_model is None:
            self.build_cluster_model()

        if method == 'cosine' or method == 'hybrid':
            # Transform query to TF-IDF space
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            
            # Get top similar products
            top_indices = similarities.argsort()[-n:][::-1]
            recommendations = []
            
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include if there's some similarity
                    rec = {
                        'asin': self.product_data.iloc[idx]['asin'],
                        'title': self.product_data.iloc[idx]['title'],
                        'price': float(self.product_data.iloc[idx]['price'])
                    }
                    recommendations.append(rec)
            
            if method == 'hybrid' and self.cluster_model is not None:
                # Add cluster-based filtering
                cluster_recs = self.get_recommendations(query, method='cluster', n=n)
                recommendations.extend(cluster_recs)
                # Remove duplicates and sort by relevance
                recommendations = sorted(list({r['asin']: r for r in recommendations}.values()),
                                      key=lambda x: similarities[self.product_data.index[self.product_data['asin'] == x['asin']].tolist()[0]], 
                                      reverse=True)[:n]
            
            return recommendations

        elif method == 'cluster':
            if self.cluster_model is None:
                return []
            
            # Transform query to TF-IDF space
            query_vec = self.tfidf_vectorizer.transform([query])
            cluster_pred = self.cluster_model.predict(query_vec)[0]
            
            # Get products from the same cluster
            cluster_products = self.product_data[self.product_data['cluster'] == cluster_pred]
            return [{'asin': row['asin'], 'title': row['title'], 'price': float(row['price'])} 
                    for _, row in cluster_products.sample(min(n, len(cluster_products))).iterrows()]

        return []
    
    def get_recommendations_from_audio(self, audio_file_path, method='cosine', n=5):
        """Transcribe audio query and get product recommendations"""
        try:
            # Transcribe audio to text
            query = transcribe_audio(audio_file_path)
            print(f"Audio query transcribed: '{query}'")
            
            # Get recommendations based on transcribed text
            return self.get_recommendations(query, method=method, n=n)
        except Exception as e:
            print(f"Error processing audio query: {str(e)}")
            return []

    def evaluate_model(self, n_queries=100, methods=['cosine', 'cluster', 'hybrid']):
        print(f"Evaluating recommendation model on {n_queries} random queries...")
        test_queries = self.query_generator.generate_queries(n_queries)
        results = {'queries': test_queries, 'recommendations': {}, 'stats': {}}
        
        for method in methods:
            results['recommendations'][method] = []
            results['stats'][method] = {'avg_recommendations': 0, 'query_success_rate': 0, 'avg_time': 0}
        
        for method in methods:
            total_recs = 0
            successful_queries = 0
            total_time = 0
            
            for query in test_queries:
                start_time = time.time()
                recs = self.get_recommendations(query, method=method)
                elapsed_time = time.time() - start_time
                
                results['recommendations'][method].append({
                    'query': query,
                    'results': recs,
                    'time': elapsed_time
                })
                
                total_time += elapsed_time
                if recs:
                    successful_queries += 1
                    total_recs += len(recs)
            
            results['stats'][method]['avg_recommendations'] = total_recs / n_queries
            results['stats'][method]['query_success_rate'] = successful_queries / n_queries
            results['stats'][method]['avg_time'] = total_time / n_queries
            
        return results

    def run_comprehensive_tests(self, queries_per_type=10):
        query_types = {
            'basic': ["{category}"],
            'price_filtered': ["{category} under ${price}", "affordable {category}"],
            'bestseller': ["bestselling {category}", "popular {category}"],
            'rating': ["top rated {category}", "highly rated {category}", "{category} with good reviews"],
            'specific_features': ["{adjective} {category}", "{color} {category}"],
            'brand_specific': ["{brand} {category}", "alternatives to {brand} {category}"],
            'complex': ["bestselling {category} under ${price}", "{category} between ${price_min} and ${price_max}"]
        }
        
        results = {}
        original_templates = self.query_generator.query_templates
        
        for query_type, templates in query_types.items():
            print(f"Testing {query_type} queries...")
            self.query_generator.query_templates = templates
            type_queries = self.query_generator.generate_queries(queries_per_type)
            
            type_results = {'queries': type_queries, 'cosine': [], 'cluster': [], 'hybrid': []}
            
            for query in type_queries:
                for method in ['cosine', 'cluster', 'hybrid']:
                    recs = self.get_recommendations(query, method=method)
                    type_results[method].append({'query': query, 'count': len(recs), 'results': recs})
            
            results[query_type] = type_results
        
        self.query_generator.query_templates = original_templates
        return results

    def batch_test_and_save(self, output_file='recommendation_test_results.csv', n_queries=500):
        print(f"Running batch test on {n_queries} queries...")
        test_queries = self.query_generator.generate_queries(n_queries)
        results = []
        
        for i, query in enumerate(test_queries):
            if i % 50 == 0:
                print(f"Processing query {i+1}/{n_queries}...")
                
            for method in ['cosine', 'cluster', 'hybrid']:
                start_time = time.time()
                recs = self.get_recommendations(query, method=method)
                elapsed_time = time.time() - start_time
                
                result = {
                    'query': query,
                    'method': method,
                    'num_results': len(recs),
                    'processing_time': elapsed_time,
                    'success': len(recs) > 0
                }
                
                if recs:
                    result.update({
                        'first_result_asin': recs[0].get('asin', ''),
                        'first_result_title': recs[0].get('title', ''),
                        'first_result_price': recs[0].get('price', 0)
                    })
                else:
                    result.update({
                        'first_result_asin': '',
                        'first_result_title': '',
                        'first_result_price': 0
                    })
                
                results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return results_df

if __name__ == "__main__":
    # Create sample data (replacing load_amazon_data)
    print("Creating sample Amazon data for demonstration.")
    categories = ['Electronics', 'Books', 'Home & Kitchen', 'Clothing', 'Sports & Outdoors', 
                  'Beauty', 'Toys & Games', 'Grocery', 'Pet Supplies', 'Automotive']
    brands = ['Amazon', 'Apple', 'Samsung', 'Sony', 'LG', 'Bose', 'Nike', 'Adidas', 
              'Logitech', 'Microsoft', 'Dell', 'HP', 'Anker', 'JBL', 'Canon', 'Nikon']
    
    n_samples = 2000
    titles = []
    for _ in range(n_samples):
        brand = random.choice(brands)
        category = random.choice(categories)
        product_type = random.choice(['Pro', 'Ultra', 'Max', 'Premium', 'Basic', 'Plus', 'Lite', ''])
        model = random.choice(['X', 'S', 'A', 'Z', 'M', 'Q', 'V']) + str(random.randint(1, 100))
        titles.append(f"{brand} {category.split()[0]} {product_type} {model}".strip())
    
    amazon_data = pd.DataFrame({
        'asin': [f'B{i:09d}' for i in range(1, n_samples+1)],
        'title': titles,
        'description': [f"This is a great product with many features." for _ in range(n_samples)],
        'category': np.random.choice(categories, n_samples),
        'price': np.random.uniform(10, 500, n_samples).round(2),
        'rating': np.random.uniform(1, 5, n_samples).round(1),
        'review_count': np.random.randint(0, 2000, n_samples),
        'sales_rank': np.random.randint(1, 500000, n_samples)
    })
    
    # Initialize and run recommender
    recommender = AmazonProductRecommender(amazon_data)
    print("Building recommendation models...")
    recommender.build_cosine_model()
    recommender.build_cluster_model()
    
    print("\nTesting with generated queries:")
    generated_queries = recommender.query_generator.generate_queries(5)
    for query in generated_queries:
        print(f"\nQuery: {query}")
        recommendations = recommender.get_recommendations(query, method='hybrid')
        print("Recommendations:")
        for i, rec in enumerate(recommendations[:3]):
            print(f"{i+1}. {rec['title']} - ${rec['price']:.2f}")
    
    # Demonstrate audio query capability (comment out if no audio file available)
    # audio_file = "sample_query.mp3"  # Replace with actual audio file path
    # if os.path.exists(audio_file):
    #     print(f"\nTesting with audio query from {audio_file}:")
    #     audio_recommendations = recommender.get_recommendations_from_audio(audio_file, method='hybrid')
    #     print("Recommendations from audio query:")
    #     for i, rec in enumerate(audio_recommendations[:3]):
    #         print(f"{i+1}. {rec['title']} - ${rec['price']:.2f}")
    
    print("\nRunning comprehensive evaluation...")
    eval_results = recommender.evaluate_model(n_queries=20)
    
    print("\nModel Evaluation Summary:")
    for method, stats in eval_results['stats'].items():
        print(f"\nMethod: {method.upper()}")
        print(f"Average recommendations per query: {stats['avg_recommendations']:.2f}")
        print(f"Query success rate: {stats['query_success_rate']*100:.1f}%")
        print(f"Average processing time: {stats['avg_time']*1000:.2f}ms")
    
    print("\nRunning targeted query tests...")
    type_results = recommender.run_comprehensive_tests(queries_per_type=3)
    
    print("\nSuccess Rates by Query Type:")
    for query_type, results in type_results.items():
        success_rates = {}
        for method in ['cosine', 'cluster', 'hybrid']:
            successful = sum(1 for item in results[method] if item['count'] > 0)
            success_rates[method] = successful / len(results[method]) * 100
        print(f"{query_type.capitalize()} Queries:")
        for method, rate in success_rates.items():
            print(f"  - {method.capitalize()}: {rate:.1f}%")
    
    print("\nRunning large batch test...")
    batch_results = recommender.batch_test_and_save(n_queries=50)
    
    print("\nBatch Test Summary:")
    method_stats = batch_results.groupby('method').agg({
        'num_results': 'mean',
        'processing_time': 'mean',
        'success': 'mean'
    })
    print(method_stats)