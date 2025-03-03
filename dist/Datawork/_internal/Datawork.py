import pandas as pd
import matplotlib.pyplot as plt
import logging
import tkinter as tk
from tkinter import simpledialog, messagebox
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from ImportScrapy import run_scraper

# Configure logging
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Run the web scraper before processing data
try:
    run_scraper()
    print("Scraper execution completed. Proceeding with data analysis...")
except Exception as e:
    logging.error(f"Error executing the scraper: {e}")
    print(f"Error executing the scraper: {e}")

# Load the scraped dataset
df = pd.read_csv('books_data.csv', delimiter=';')

# Generate a word cloud from book titles
text_corpus = ' '.join(df['Title'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Book Titles")
plt.show()

# Prepare data for clustering
scaler = StandardScaler()
df['Price'] = df['Price'].str.replace(r'[^0-9.]', '', regex=True).astype(float)  # Remove currency symbols
features = df[['Price', 'Rating']]
scaled_features = scaler.fit_transform(features)

# Determine optimal cluster count using Elbow Method and Silhouette Score
wcss = []
silhouette_scores = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, labels))

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal K")
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal K")
plt.show()

# Suggest optimal K value
optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Suggested optimal number of clusters: {optimal_k}")

# Apply K-Means Clustering with optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Plot Clustering Results with Cluster Centers
plt.figure(figsize=(8, 6))
plt.scatter(df['Price'], df['Rating'], c=df['Cluster'], cmap='viridis', alpha=0.6, label="Data Points")
plt.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
            kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
            c='red', marker='X', s=200, label="Cluster Centers")
plt.xlabel("Price")
plt.ylabel("Rating")
plt.title(f"K-Means Clustering of Books (K={optimal_k})")
plt.colorbar(label="Cluster")
plt.legend()
plt.show()

# Create a simple UI to get MLP parameters with validation
root = tk.Tk()
root.withdraw()

def get_validated_integer(prompt, min_value, max_value):
    while True:
        value = simpledialog.askinteger("MLP Config", prompt, minvalue=min_value, maxvalue=max_value)
        if value is not None:
            return value
        messagebox.showerror("Input Error", f"Please enter a number between {min_value} and {max_value}.")

def get_validated_activation():
    valid_activations = {"relu", "tanh", "logistic"}
    while True:
        activation = simpledialog.askstring("MLP Config", "Enter activation function (relu, tanh, logistic):").strip().lower()
        if activation in valid_activations:
            return activation
        messagebox.showerror("Input Error", "Invalid activation function. Choose from relu, tanh, logistic.")

hidden_layer = get_validated_integer("Enter number of hidden layer neurons:", 1, 200)
activation_function = get_validated_activation()
max_iter = get_validated_integer("Enter maximum iterations:", 100, 1000)

# Train an MLP Classifier
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['Cluster'], test_size=0.2, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer,), activation=activation_function, solver='adam', max_iter=max_iter, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
logging.info(f'MLP Classification Accuracy: {accuracy:.2f}')
print(f'MLP Classification Accuracy: {accuracy:.2f}')

# Plot MLP Classification Results with Accuracy
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', alpha=0.6, label="Predicted Data")
plt.xlabel("Price (Scaled)")
plt.ylabel("Rating (Scaled)")
plt.title(f"MLP Classification Results (Accuracy: {accuracy:.2f})")
plt.colorbar(label="Predicted Cluster")
plt.legend()
plt.show()