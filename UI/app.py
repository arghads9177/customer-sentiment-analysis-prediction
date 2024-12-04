import streamlit as st
import pandas as pd
import nltk
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from textblob import TextBlob
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def load_credentials():
    with open("../data/admin_credentials.json") as f:
        return json.load(f)


# Utility functions
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)

def plot_class_distribution(data):
    fig, ax = plt.subplots()
    sns.countplot(x='Sentiment', data=data, ax=ax, palette='Set2')
    ax.set_title('Class Distribution')
    st.pyplot(fig)

def generate_wordcloud(data, sentiment):
    reviews = data[data['Sentiment'] == sentiment]['Review']
    combined_text = " ".join(reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def plot_sentiment_score_distribution(data):
    data['polarity'] = data['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    fig, ax = plt.subplots()
    sns.histplot(data['polarity'], bins=30, kde=True, ax=ax, color='skyblue')
    ax.set_title('Sentiment Polarity Distribution')
    st.pyplot(fig)

def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'negative'])
    disp.plot(cmap='Blues')
    st.pyplot(plt.gcf())
    
def link_style_menu():
    st.sidebar.markdown(
        """
        <style>
        .sidebar-link {
            font-size: 18px;
            color: #007BFF;
            text-decoration: none;
            margin-bottom: 10px;
            display: block;
        }
        .sidebar-link:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    menu = {
        "Sentiment Prediction": "Sentiment Prediction",
        "Sentiment Class Analysis": "Sentiment Class Analysis",
        "Word Clouds": "Word Clouds",
        "Sentiment Distribution by Product": "Sentiment Distribution by Product"
    }
    
    # Display link-style menu
    selected_option = st.sidebar.radio(
        "",
        options=list(menu.values()),
        label_visibility="collapsed"
    )
    
    return selected_option

# Streamlit app
def main():
    st.title("Customer Sentiment Analysis")
    # menu = ["User Section", "Admin Section", "Data Insights"]
    # choice = st.sidebar.selectbox("Select Section", menu)
    choice = link_style_menu()

    # Sentiment Prediction
    if choice == "Sentiment Prediction":
        st.subheader("User - Sentiment Prediction")
        review_input = st.text_area("Enter a customer review:")
        if st.button("Predict Sentiment"):
            try:
                with open("../models/saved_model.pkl", "rb") as model_file, open("../models/vectorizer.pkl", "rb") as vectorizer_file:
                    model = pickle.load(model_file)
                    vectorizer = pickle.load(vectorizer_file)
                cleaned_review = preprocess_text(review_input)
                vectorized_review = vectorizer.transform([cleaned_review])
                prediction = model.predict(vectorized_review)
                sentiment = ["negative", "neutral", "positive"][prediction[0]]

                st.write(f"Predicted Sentiment: {sentiment.capitalize()}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
       

    # Sentiment Class Analysis
    elif choice == "Sentiment Class Analysis":
        st.subheader("Sentiment Class Distribution")
        data_path = "../data"
        csv_path = os.path.join(data_path, "Cleaned_SA.csv")
        data = pd.read_csv(csv_path)
        plot_class_distribution(data)
        # Title
        st.markdown("## **Class Distribution Overview**")
        
        # Positive Sentiments
        st.markdown("""
        #### Positive Sentiments:
        - There are **122,808 positive** reviews, which dominate the dataset, accounting for approximately **80.6%** of the total reviews.
        - This suggests that customers mostly have favorable opinions about the product/service.
        """)
        
        # Negative Sentiments
        st.markdown("""
        #### Negative Sentiments:
        - There are **23,353 negative** reviews, representing about **15.3%** of the dataset.
        - While less frequent than positive sentiments, this group is critical for identifying pain points and areas for improvement.
        """)
        
        # Neutral Sentiments
        st.markdown("""
        #### Neutral Sentiments:
        - There are **8,306 neutral** reviews, making up approximately **5.5%** of the dataset.
        - Neutral reviews might include comments that are factual or lack emotional tone.
        """)
        
        # Imbalance in Sentiment Classes
        st.markdown("## **Imbalance in Sentiment Classes**")
        st.markdown("""
        - Positive reviews heavily outweigh negative and neutral ones.
        - This imbalance could lead to biased model predictions where the classifier favors the positive class.
        
        #### Class Ratios:
        - **Positive:Negative** ratio = **~5:1**
        - **Positive:Neutral** ratio = **~15:1**
        """)
        
        # Business Implications
        st.markdown("## **Business Implications**")
        
        # Strengths
        st.markdown("""
        #### Strengths:
        - The high proportion of positive sentiments reflects strong customer satisfaction, likely due to quality products/services, effective customer support, or good value.
        """)
        
        # Challenges
        st.markdown("""
        #### Challenges:
        - The negative sentiments, while smaller in volume, require careful attention to address customer pain points and improve retention.
        - Neutral sentiments might represent missed opportunities for stronger customer engagement or satisfaction.
        """)
        
        # Recommendations
        st.markdown("## **Recommendations**")
        
        # Address Imbalance for Model Training
        st.markdown("""
        ### 1. **Address Imbalance for Model Training**
        - **Use Oversampling/Undersampling:**
            - Employ techniques like SMOTE (Synthetic Minority Oversampling Technique) for the minority classes (negative and neutral) to balance the dataset.
        - **Class Weighting:**
            - Use class weights in the model to give more importance to the minority classes.
        """)
        
        # Focus on Negative Sentiments
        st.markdown("""
        ### 2. **Focus on Negative Sentiments**
        - **Root Cause Analysis:**
            - Identify recurring themes in negative reviews using topic modeling (e.g., LDA) or keyword extraction.
            - Investigate common issues like product defects, customer service complaints, or delays.
        - **Business Strategy:**
            - Address these issues promptly to improve customer satisfaction and retention.
        """)
        
        # Leverage Neutral Reviews
        st.markdown("""
        ### 3. **Leverage Neutral Reviews**
        - Neutral reviews often provide feedback without explicit sentiment.
        - Analyze them for actionable insights like feature requests, minor grievances, or areas for improvement.
        """)

    # Word Clouds
    elif choice == "Word Clouds":
        st.subheader("Word Clouds Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "Cleaned_SA.csv")
        data = pd.read_csv(csv_path)
        st.write("Positive Reviews")
        generate_wordcloud(data, 'positive')
        st.write("Negative Reviews")
        generate_wordcloud(data, 'negative')
        st.write("Neutral Reviews")
        generate_wordcloud(data, 'neutral')
        #uploaded_file = st.file_uploader("Upload Dataset for Analysis (CSV)", type=["csv"])
        # Positive Sentiment
        st.markdown("## **Positive Sentiment**")
        st.markdown("""
        - **Key Words Identified:**
            **"Great," "Perfect," "Worth," "Every Penny," "Terrific Purchase," "Must Buy"**
        
        ### Insights:
        - Customers highly value the product/service quality and find it to be worth the price ("every penny").
        - Terms like "great" and "perfect" suggest an exceptional customer experience.
        - Words like "terrific purchase" and "must buy" indicate strong satisfaction and recommendations, likely leading to repeat purchases or positive referrals.
        
        ### Business Takeaways:
        - Highlight positive phrases like these in marketing campaigns to attract new customers.
        - Reinforce the factors contributing to positive feedback, such as product quality or affordability.
        """)
        
        # Negative Sentiment
        st.markdown("## **Negative Sentiment**")
        st.markdown("""
        - **Key Words Identified:**
            **"Money," "Waste," "Absolute Rubbish," "Worst Experience," "Utterly Disappointed," "Terrible Product"**
        
        ### Insights:
        - Negative sentiments revolve around a sense of financial loss or dissatisfaction with the product ("money," "waste").
        - Terms like "absolute rubbish" and "terrible product" reflect extreme dissatisfaction, possibly due to product quality issues.
        - "Worst experience" and "utterly disappointed" highlight problems with service, product expectations, or usability.
        
        ### Business Takeaways:
        - Investigate and address recurring complaints related to product quality or misleading expectations.
        - Introduce proactive measures such as improved product descriptions, warranties, or better customer service.
        - Prioritize resolving issues contributing to financial dissatisfaction (e.g., poor returns policy).
        """)
        
        # Neutral Sentiment
        st.markdown("## **Neutral Sentiment**")
        st.markdown("""
        - **Key Words Identified:**
            **"Job," "Good," "Okay," "Decent Product," "Specified," "Fair"**
        
        ### Insights:
        - Neutral sentiments reflect moderate experiences where the product met expectations but did not exceed them.
        - Words like "okay" and "good" suggest general satisfaction without enthusiasm.
        - "Fair" and "decent product" indicate that the product's quality aligns with expectations, neither excelling nor disappointing.
        
        ### Business Takeaways:
        - Analyze neutral feedback for opportunities to enhance product features or customer experience.
        - Offer incentives, such as discounts or loyalty programs, to engage these customers and convert them into advocates.
        """)
        
        # General Observations
        st.markdown("## **General Observations**")
        st.markdown("""
        ### Sentiment Distribution:
        - Positive sentiment dominates, indicating overall customer satisfaction.
        - Negative and neutral reviews offer valuable insights for improving products and processes.
        
        ### Actionable Insights:
        - Use positive reviews to identify strengths for marketing.
        - Investigate negative feedback to resolve root causes of dissatisfaction.
        - Leverage neutral reviews to identify incremental improvements.
        """)
        
    # Sentiment Distribution by Product
    elif choice == "Sentiment Distribution by Product":
        st.subheader("Top Product with Negative Feedback")
        data_path = "../data"
        csv_path = os.path.join(data_path, "Cleaned_SA.csv")
        data = pd.read_csv(csv_path)
        product_sentiment = data[["product_name", "Sentiment"]].groupby(["product_name", "Sentiment"]).value_counts().unstack().reset_index()
        product_sentiment.fillna(0, inplace= True)
        product_sentiment["total"] = product_sentiment["negative"] + product_sentiment["neutral"] + product_sentiment["positive"]
        product_sentiment["positive_percentage"] = product_sentiment["positive"] / product_sentiment["total"] * 100
        product_sentiment["negative_percentage"] = product_sentiment["negative"] / product_sentiment["total"] * 100
        st.write(product_sentiment[["product_name", "negative_percentage"]].sort_values("negative_percentage", ascending= False).head(10))
        fig, ax= plt.subplots()
        sns.barplot(data = product_sentiment[["product_name", "negative_percentage"]].sort_values("negative_percentage", ascending= False).head(10),
           x = "product_name", y= "negative_percentage", hue= "product_name", ax= ax)
        ax.set_title("Top Product with Negative Feedback")
        plt.xlabel("Product Name")
        plt.ylabel("Negative(%)")
        plt.xticks(rotation= 90)
        st.pyplot(fig)
        st.markdown("#### High Negative Feedback Percentages:")
        st.markdown("""
        - Several products have a significant proportion of negative feedback, with percentages ranging from **56% to 75%**.
        - The **Sai Store Light Blue Cotton Carpet** leads with a **75% negative feedback**, indicating severe dissatisfaction among customers.
        """)
        
        st.markdown("#### Product Categories Affected:")
        st.markdown("""
        - Products span across various categories, including:
            - **Home Appliances** *(e.g., "MAHARAJA WHITELINE Fortune FP - 102," "HAVELLS convenio")*.
            - **Electronics** *(e.g., "realme 4k Smart Google TV Stick," "TXOR NEXUS Smartwatch")*.
            - **Office Supplies** *(e.g., "PrintStar 12A Compatible Toner Cartridge")*.
            - **Household Items** *(e.g., "Aquagrand Water Purifier," "Sai Store Cotton Carpet")*.
        """)
        
        st.markdown("#### Potential Issues:")
        st.markdown("""
        - **Product Quality:** The high percentage of negative reviews may reflect product defects, poor durability, or failure to meet customer expectations.
        - **User Experience:** Products such as smartwatches and TV sticks might suffer from usability issues or misleading marketing.
        - **Performance Problems:** Home appliances and electronics could be underperforming or not living up to advertised standards.
        - **Product Description Mismatch:** Items like "BALKRISHNA ENTERPRISE Green Tea Sticks" might have unclear descriptions or fail to meet buyer expectations.
        """)
        
        st.markdown("#### Business Implications:")
        st.markdown("""
        - **Customer Retention:** High negative feedback can lead to customer churn and damage brand reputation.
        - **Brand Perception:** Frequent complaints about specific products can harm trust in the overall brand.
        - **Sales Impact:** Negative feedback likely impacts sales volume and revenue for these products.
        """)
        
        st.markdown("#### Recommendations for Improvement:")
        st.markdown("""
        - **Root Cause Analysis:**
            - Investigate specific complaints for these products to understand recurring issues.
        - **Product Quality Enhancement:**
            - Collaborate with manufacturers to improve product durability and performance.
        - **Customer Support & Resolution:**
            - Enhance after-sales support for affected products to address grievances and rebuild trust.
        - **Revised Product Listings:**
            - Update product descriptions to better align with actual features and performance.
        - **Targeted Monitoring:**
            - Closely monitor these products' reviews to track improvements or persistent issues after changes.
        """)

#Calling the main function
if __name__ == "__main__":
    main()