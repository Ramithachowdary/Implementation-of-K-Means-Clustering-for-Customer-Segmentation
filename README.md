# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries
2. Load and Explore the Dataset
3. Select Features for Clustering
4. Calculate WCSS (Within-Cluster Sum of Squares)
5. Plot the Elbow Graph
6. Apply K-Means Clustering with Optimal Clusters
7. Segment Data Based on Clusters
8. Visualize the Clusters
9. Analyze and Interpret Results
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Ramitha chowdary s 
RegisterNumber:24900704


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Display the first few rows
print(data.head())

# Display dataset information
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Initialize Within-Cluster Sum of Squares (WCSS)
wcss = []
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")
warnings.filterwarnings("ignore", category=FutureWarning)
# Elbow Method to find the optimal number of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Apply K-Means with the chosen number of clusters
km = KMeans(n_clusters=5)
km.fit(data.iloc[:, 3:])

# Predict the clusters
y_pred = km.predict(data.iloc[:, 3:])
data["cluster"] = y_pred

# Segment the data based on clusters
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

# Plot the customer segments
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="blue", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="green", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="purple", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")
plt.legend()
plt.title("Customer Segments")
plt.show()

 
*/
```

## Output:
![K Means Clustering for Customer Segmentation](sam.png)
```
 CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   CustomerID              200 non-null    int64 
 1   Gender                  200 non-null    object
 2   Age                     200 non-null    int64 
 3   Annual Income (k$)      200 non-null    int64 
 4   Spending Score (1-100)  200 non-null    int64 
dtypes: int64(4), object(1)
memory usage: 7.9+ KB
None
CustomerID                0
Gender                    0
Age                       0
Annual Income (k$)        0
Spending Score (1-100)    0
dtype: int64
```
![download](https://github.com/user-attachments/assets/d97b801b-5b24-49e7-8a04-e53bf933d672)
![download](https://github.com/user-attachments/assets/fe82602e-6c6e-4ba5-be16-cdae691578c3)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
