import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('iris.csv')

# # Summary statistics
# print(df.describe())

# # Histograms
# df.hist(figsize=(10,6))
# plt.show()

# # Boxplot
# sns.boxplot(data=df)
# plt.show()

# Correlation matrix
numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
