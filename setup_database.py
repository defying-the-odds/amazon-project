import sqlite3
import pandas as pd

# Step 1: Connect to SQLite database
# Creates 'products.db' if it doesn’t exist
conn = sqlite3.connect("products.db")
cursor = conn.cursor()

# Step 2: Create the products table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        size TEXT,
        description TEXT
    )
''')
print("Table 'products' created or already exists.")

# Step 3: Load the cleaned CSV into a DataFrame
df = pd.read_csv("merged_with_category_cleaned.csv")
print("Loaded cleaned_data.csv:")
print(df.head())  # Preview the data

# Step 4: Insert DataFrame into SQLite table
# 'if_exists="replace"' overwrites the table if it already exists
df.to_sql("products", conn, if_exists="replace", index=False)
print("Data inserted into 'products' table.")

# Step 5: Verify the data was loaded
cursor.execute("SELECT * FROM products LIMIT 5")
rows = cursor.fetchall()
print("Sample data from database:")
for row in rows:
    print(row)

# Step 6: Commit changes and close the connection
conn.commit()
conn.close()
print("Database setup complete! File saved as 'products.db'.")