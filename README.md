import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import smtplib
from email.message import EmailMessage
import logging

# Setup logging
logging.basicConfig(filename='marketing_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# --- 1. Sample Data ---

# Customer Data
customer_data = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['John', 'Alice', 'Bob', 'Mary', 'Tom'],
    'email': ['john@example.com', 'alice@example.com', 'bob@example.com', 'mary@example.com', 'tom@example.com'],
    'age': [25, 30, 35, 40, 45],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'annual_income': [30000, 50000, 40000, 60000, 35000],
    'purchase_history': [1000, 1500, 1200, 2000, 1100]
})

# Product Data
product_data = pd.DataFrame({
    'product_id': [101, 102, 103, 104, 105],
    'product_name': ['Luxury Watch', 'Budget Phone', 'Smart TV', 'Premium Headphones', 'Fashion Bag'],
    'category': ['Luxury', 'Electronics', 'Electronics', 'Luxury', 'Fashion']
})

# Encode categorical features
customer_data['gender'] = customer_data['gender'].map({'M': 0, 'F': 1})

# --- 2. Customer Segmentation ---

features = ['age', 'gender', 'annual_income', 'purchase_history']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data[features])

kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['segment'] = kmeans.fit_predict(X_scaled)

# --- 3. Personalized Marketing Logic ---

def get_marketing_message(segment):
    messages = {
        0: "Discover exclusive luxury deals hand-picked for you!",
        1: "Special savings on top electronics—shop now!",
        2: "Fresh fashion arrivals you can't miss!"
    }
    return messages.get(segment, "Thank you for being a valued customer!")

def recommend_products(segment):
    segment_preferences = {
        0: 'Luxury',
        1: 'Electronics',
        2: 'Fashion'
    }
    category = segment_preferences.get(segment, 'Electronics')
    return product_data[product_data['category'] == category]

# --- 4. Email Sending Stub ---

def send_email(to_email, subject, body):
    # NOTE: This is a stub. Replace with real credentials and SMTP setup.
    print(f"Sending email to {to_email}:\nSubject: {subject}\n{body}\n")
    # Uncomment and configure the below to send real emails.
    # msg = EmailMessage()
    # msg.set_content(body)
    # msg['Subject'] = subject
    # msg['From'] = 'your_email@example.com'
    # msg['To'] = to_email
    # with smtplib.SMTP('smtp.example.com', 587) as server:
    #     server.starttls()
    #     server.login('your_email@example.com', 'password')
    #     server.send_message(msg)

# --- 5. Personalized Campaign Execution ---

def run_campaign():
    for _, customer in customer_data.iterrows():
        segment = customer['segment']
        message = get_marketing_message(segment)
        recommendations = recommend_products(segment)

        email_body = f"""
Hi {customer['name']},

{message}

We recommend these products for you:
- {', '.join(recommendations['product_name'].tolist())}

Visit our store to explore more!

Best,
Your Company
"""
        send_email(customer['email'], "Personalized Offers Just for You!", email_body)

        # Log interaction
