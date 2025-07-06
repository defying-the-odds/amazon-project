import pandas as pd

# Load both CSVs
file1 = pd.read_csv("amazon_categories.csv")  # Contains 'id' and 'category'
file2 = pd.read_csv("amazon_products.csv")  # Contains 'id' and 10 other columns

# Merge them based on the 'id' column
merged = pd.merge(file2, file1, left_on='category_id', right_on='id', how='left')

# Drop the redundant 'id' column (from file1)
merged.drop(columns=['id'], inplace=True)

# Save the result
merged.to_csv("merged_with_category.csv", index=False)
