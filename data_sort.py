import pandas as pd

# Step 1: Read the CSV file
file_path = "merged_with_category.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Step 2: Remove duplicate rows
df = df.drop_duplicates()

# Step 3: Replace missing numbers with 0.0 and missing strings with ""
df = df.fillna({col: 0.0 if df[col].dtype in ['int64', 'float64'] else "" for col in df.columns})


# Check if there are any missing values in the DataFrame
missing_values = df.isnull().sum().sum()

if missing_values == 0:
    print("✅ No missing values found in the dataset.")
else:
    print(f"⚠️ Warning: {missing_values} missing values still exist.")


# Define the output file name
output_file = "erged_with_category_cleaned.csv"

# Save the cleaned DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"✅ Cleaned data has been saved to '{output_file}'.")

print(df.info())

