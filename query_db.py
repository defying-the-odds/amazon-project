import sqlite3

def get_products(query_params, limit=5):
    """
    Query the products database and return top recommendations based on user preferences.
    
    Args:
        query_params (dict): Dictionary with keys like 'keywords', 'price', 'stars', etc.
                             Example: {"keywords": "running shoes", "price": 100.0}
        limit (int): Maximum number of results to return (default: 5).
    
    Returns:
        list: List of tuples, top 'limit' product rows sorted by relevance.
    """
    # Connect to the database
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()

    # Build the SQL query dynamically
    sql = "SELECT * FROM products WHERE 1=1"
    params = []

    if "keywords" in query_params:
        sql += " AND title LIKE ?"
        params.append(f"%{query_params['keywords']}%")
    if "price" in query_params:
        sql += " AND price <= ?"
        params.append(query_params["price"])
    if "stars" in query_params:
        sql += " AND stars >= ?"
        params.append(query_params["stars"])
    if "reviews" in query_params:
        sql += " AND reviews >= ?"
        params.append(query_params["reviews"])
    if "category_id" in query_params:
        sql += " AND category_id = ?"
        params.append(query_params["category_id"])
    if "isBestSeller" in query_params:
        sql += " AND isBestSeller = ?"
        params.append(1 if query_params["isBestSeller"] else 0)
    if "boughtInLastMonth" in query_params:
        sql += " AND boughtInLastMonth >= ?"
        params.append(query_params["boughtInLastMonth"])

    # Sort by stars (descending) and boughtInLastMonth (descending), then limit
    sql += " ORDER BY stars DESC, boughtInLastMonth DESC LIMIT ?"
    params.append(limit)

    # Execute and fetch results
    cursor.execute(sql, params)
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    return results

# Test the function
if __name__ == "__main__":
    test_cases = [
        {"keywords": "running shoes", "price": 100.0, "stars": 4.0},
        {"keywords": "jacket", "reviews": 50},
        {"category_id": 5, "isBestSeller": True},
        {"boughtInLastMonth": 100}
    ]

    for test in test_cases:
        print(f"\nQuerying with: {test}")
        products = get_products(test, limit=5)  # Limit to 5
        if products:
            for product in products:
                print(product)
        else:
            print("No matching products found.")