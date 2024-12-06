#Importing Required Libraries
#I imported the necessary libraries for data handling, scaling, clustering, and visualization
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

#1. Loading the Dataset
#I loaded the customer data from a CSV file
data = pd.read_csv("chapter_2/mall_customers.csv")

#2. Selecting Features
#I selected the 'Annual Income' and 'Spending Score' features for clustering
features = data[["Annual Income (k$)", "Spending Score (1-100)"]]

#3. Scaling Features
#I standardized the selected features to improve clustering performance
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#4. Determining the Optimal Number of Clusters
#I used the Elbow Method to identify the optimal number of clusters by plotting inertia values
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=53)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

#I plotted the results of the Elbow Method
plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Determining the Number of Clusters with the Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Errors)")
plt.get_current_fig_manager().set_window_title("Elbow Method")
plt.show()

#5. Applying K-Means Clustering
#I set the optimal number of clusters based on the Elbow Method
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=53)
#I assigned cluster labels to each data point
clusters = kmeans.fit_predict(features_scaled)

#I added the cluster labels to the original dataset
data['Cluster'] = clusters

#Calculating the Silhouette Score
#I calculated the Silhouette Score to measure the quality of the clustering
silhouette_avg = silhouette_score(features_scaled, clusters)
print(f"\nSilhouette Score for k={optimal_k}: {silhouette_avg}")

#6. Visualizing Clustering Results
#I plotted the clustered data points with their respective cluster assignments
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=features_scaled[:, 0], 
    y=features_scaled[:, 1], 
    hue=data['Cluster'], 
    palette='viridis', 
    s=50
)
plt.title("K-Means Clustering Results")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.legend(title="Clusters")
plt.get_current_fig_manager().set_window_title("K-Means Clustering")
plt.show()