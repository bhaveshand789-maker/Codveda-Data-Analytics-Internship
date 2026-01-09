import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('iris.csv')

# Scatter plot
sns.scatterplot(x="sepal_length", y="petal_length", data=df)
plt.title("Sepal vs Petal Length")
plt.show()

# Bar plot
df['species'].value_counts().plot(kind='bar')
plt.title("Species Count")
plt.show()
