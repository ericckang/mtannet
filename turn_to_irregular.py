import pandas as pd

# Load the CSV file with the correct delimiter and decimal settings
df = pd.read_csv("AirQualityUCI.csv", delimiter=';', decimal=',')

# Print the first few rows to inspect the column names
print(df.head())

# Check if 'Date' and 'Time' columns exist and have no leading/trailing spaces
df.columns = df.columns.str.strip()

# Convert 'Date' and 'Time' columns to a single 'Datetime' column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')

# Drop the original 'Date' and 'Time' columns
df.drop(columns=['Date', 'Time'], inplace=True)

# Set the 'Datetime' column as the index
df.set_index('Datetime', inplace=True)

# Simulate irregular sampling by randomly dropping 50% of the rows
df_irregular = df.sample(frac=0.5, random_state=42).sort_index()

# Save the irregular dataset to a new CSV file
df_irregular.to_csv("AirQualityUCI_Irregular.csv")

print("Original dataset shape:", df.shape)
print("Irregular dataset shape:", df_irregular.shape)
