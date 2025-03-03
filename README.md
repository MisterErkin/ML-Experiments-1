# 📘 ML-Experiments-1: Book Data Analysis & Machine Learning

## 🔍 Overview

**ML-Experiments-1** is the first project in the **ML-Experiments** series, focusing on **web scraping, data analysis, clustering, and machine learning**. This project extracts book data from an online store, processes it, and applies **unsupervised and supervised ML techniques** to derive insights.

## 🚀 Features

- 📚 **Web Scraping**: Extracts book titles, prices, and ratings using Scrapy.
- 📊 **Data Visualization**: Generates a **word cloud** from book titles.
- 🔢 **Clustering with K-Means**: Identifies patterns in book prices & ratings.
- 📈 **Elbow & Silhouette Methods**: Determines the **optimal number of clusters**.
- 🤖 **MLP Classifier**: Trains a neural network to predict book clusters.
- 🛠 **GUI for User Input**: Allows users to customize **ML model parameters**.

## 🏗 Project Structure

```
ML-Experiments-1/
│-- Datawork.py         # Main script (ML & Analysis)
│-- ImportScrapy.py     # Web Scraper for book data
│-- dist/Datawork.exe   # Standalone executable (no Python required)
│-- books_data.csv      # Scraped dataset (auto-generated)
│-- README.md           # Project documentation 
│-- debug_log.txt       # Logs for debugging (auto-generated)
```

## ⚡ How to Run

### **Option 1: Run the EXE File (No Python Needed)**

1. Double-click to run the program.
2. Follow the **UI prompts** to configure ML models.

### **Option 2: Run from Source Code**

#### **1️⃣ Install Dependencies**

```
pip install scrapy
pip install pandas
pip install matplotlib
pip install wordcloud
pip install scikit-learn
pip install tk
pip install logging
```

#### **2️⃣ Run the ML Analysis**

```sh
python Datawork.py
```

## 📊 Machine Learning Techniques Used

### **1. Clustering (Unsupervised Learning)**

- **K-Means Algorithm**: Groups books into clusters based on **price & rating**.
- **Elbow Method & Silhouette Score**: Determines the **best cluster count (K)**.

### **2. MLP Classifier (Supervised Learning)**

- Uses a **Multi-Layer Perceptron (MLP)** to classify books.
- Allows users to set **hidden layers, activation function, and epochs**.
- Displays **accuracy score** after training.

## 🤝 Contributing

Feel free to **fork**, submit **issues**, or suggest **new ML experiments**! 🚀


