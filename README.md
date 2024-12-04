# Customer Sentiment Analysis Project  

## Project Overview  
This project aims to analyze customer sentiments based on product reviews collected from Flipkart.com. Using natural language processing (NLP) techniques and machine learning models, the project focuses on identifying customer sentiment as **positive**, **neutral**, or **negative**. The insights generated can help businesses understand customer feedback, improve their services, and make data-driven decisions for product development and marketing strategies.

---

## About the Dataset  

### Dataset Description  
The dataset comprises customer reviews of 104 product categories from Flipkart.com, including electronics, clothing, home decor, and automated systems. It contains **205,053 rows** and **6 columns**, providing detailed information about product reviews, ratings, and sentiments.

## Data Dource

This dataset is available on Kaggle in the following link:
> https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset

### Features  

| Column         | Description                                                                                     |
|-----------------|-------------------------------------------------------------------------------------------------|
| `Product_name` | Name of the product reviewed.                                                                   |
| `Product_price`| Price of the product.                                                                           |
| `Rate`         | Customer's rating of the product (on a scale of 1 to 5).                                        |
| `Review`       | Text of the customer's review for each product.                                                 |
| `Summary`      | Descriptive summary of the customer's thoughts about the product.                               |
| `Sentiment`    | Multiclass label for sentiment: **Positive**, **Negative**, or **Neutral** (derived from Summary). |

### Data Cleaning  
- Missing values in the `Review` column are handled, and `NaN` values are included for products with no review but an available `Summary`.  
- Data cleaning was performed using Python's `NumPy` and `Pandas` libraries.  
- Sentiment labeling was conducted using the **VADER model** and manually validated for accuracy.  

### Data Collection  
The dataset was obtained via **web scraping** using the `BeautifulSoup` library from Flipkart.com in December 2022.

---

## Objectives  

1. **Sentiment Analysis:**  
   - Classify customer reviews as **Positive**, **Negative**, or **Neutral** using NLP models.  

2. **Predictive Modeling:**  
   - Use features like ratings, reviews, and summaries to predict customer behavior and product preferences.  

3. **Text Classification:**  
   - Develop text classification models for tasks such as spam detection, topic classification, and intent recognition.  

4. **NLP Applications:**  
   - Train and evaluate NLP algorithms for sentiment analysis and other domains.  

5. **Customer Insights:**  
   - Extract actionable insights from customer reviews to improve customer service and product offerings.  

---

## Usage  

### Applications of the Dataset  
1. **Sentiment Analysis:**  
   Train models to classify customer sentiments for reviews and summaries.  

2. **Predictive Modeling:**  
   Predict customer behavior, purchase patterns, and product preferences based on reviews.  

3. **Text Classification:**  
   Develop models for spam detection, topic classification, and other text-based tasks.  

4. **Customer Service Insights:**  
   Identify customer complaints, issues, and suggestions to enhance service quality.  

5. **Machine Learning Evaluation:**  
   Evaluate and benchmark sentiment analysis models using this dataset.  

---

## Methodology  

### 1. **Data Understanding**  
   - Perform **Exploratory Data Analysis (EDA)** to uncover patterns and trends in the data.  
   - Assess data quality and identify missing or inconsistent values.  

### 2. **Data Preparation**  
   - Handle missing values, duplicates, and outliers.  
   - Normalize textual data and preprocess reviews for modeling (tokenization, stemming, and lemmatization).  
   - Split data into **training**, **validation**, and **test** sets.  

### 3. **Modeling**  
   - **Baseline Models:** Use models like Naive Bayes for initial benchmarking.  
   - **Advanced Models:** Train machine learning models such as Logistic Regression, SVM, and Random Forest.  
   - **Deep Learning Models:** Implement LSTMs or Transformers (e.g., BERT) for advanced text analysis.  

### 4. **Evaluation**  
   - Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.  
   - Visualize confusion matrices and ROC curves for better understanding.  

---

## Tools and Technologies  

- **Programming Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, spaCy, BeautifulSoup, TensorFlow/Keras  
- **Models:** VADER, Logistic Regression, Naive Bayes, SVM, Random Forest, LSTM, BERT  
- **Visualization Tools:** matplotlib, seaborn, plotly  

---

## Insights  

1. **Sentiment Distribution:** Majority of reviews are positive, with a smaller proportion being neutral or negative.  
2. **Product Insights:** Some categories (e.g., electronics) show higher customer satisfaction compared to others.  
3. **Customer Behavior:** Pricing and ratings significantly influence sentiment.  

---

## About the File  

- **Filename:** `Dataset-SA.csv`  
- **Size:** Approximately 3.5–4 MB  
- **Content:**  
   - **205,053 rows**  
   - **6 columns**: Product_name, Product_price, Rate, Review, Summary, Sentiment  

---

## Conclusion  

The **Customer Sentiment Analysis Project** provides actionable insights into customer feedback, enabling businesses to refine their strategies, improve product offerings, and enhance customer experiences. By leveraging advanced NLP techniques and machine learning models, this project delivers significant value for customer-centric decision-making.

---
