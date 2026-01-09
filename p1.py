import pandas as pd

# Load dataset
df = pd.read_csv('house_Prediction_Data_Set.csv')

# Check basic info
print(df.info())

# Check missing values
print(df.isnull().sum())

# Fill missing numerical values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check result
print(df.isnull().sum())
print("Final shape:", df.shape)
