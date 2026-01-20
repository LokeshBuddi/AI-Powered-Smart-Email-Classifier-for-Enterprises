import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import re
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smart Email Classifier", layout="wide")

# Download NLTK resources (quietly)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ==========================================
# 2. LOAD RESOURCES (CACHED)
# ==========================================
@st.cache_resource
def load_resources():
    # Load BERT for Categorization
    tokenizer = DistilBertTokenizerFast.from_pretrained("./saved_models/bert_category_model")
    model_bert = DistilBertForSequenceClassification.from_pretrained("./saved_models/bert_category_model")
    
    # Load ML for Urgency
    with open('./saved_models/urgency_model.pkl', 'rb') as f:
        urgency_model = pickle.load(f)
        
    with open('./saved_models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    return tokenizer, model_bert, urgency_model, vectorizer

try:
    tokenizer, model_bert, urgency_model, vectorizer = load_resources()
    st.success("System Ready: Models Loaded Successfully")
except Exception as e:
    st.error(f"Error loading models: {e}. Did you run the 'Save Models' script first?")
    st.stop()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_email(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', "", text) 
    text = re.sub(r"http\S+|www\S+", "", text)
    # Simple punctuation removal
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    processed_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(processed_words)

def predict_category(text):
    cleaned_text = clean_email(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    # Map back to labels (0: spam, 1: complaint, 2: request, 3: feedback)
    labels = {0: 'Spam', 1: 'Complaint', 2: 'Request', 3: 'Feedback'}
    return labels[pred_id]

def predict_urgency(text):
    # Hybrid Logic
    URGENT_KEYWORDS = ["urgent", "immediately", "asap", "emergency", "critical", "deadline"]
    text_lower = text.lower()
    
    # Keyword Check
    for word in URGENT_KEYWORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return "High"
            
    # ML Check
    cleaned_text = clean_email(text)
    vec_text = vectorizer.transform([cleaned_text])
    pred_id = urgency_model.predict(vec_text)[0]
    # Map back (0: Low, 1: Medium, 2: High)
    labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    return labels[pred_id]

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.title("📧 AI-Powered Smart Email Classifier")
st.markdown("Enterprise Solution for Automated Triage & Urgency Detection")

# --- SIDEBAR: ANALYTICS ---
st.sidebar.header("📊 Analytics Dashboard")

# Simulate Historical Data for Charts (Requirement: "Charts & Filters")
np.random.seed(42)
dummy_data = pd.DataFrame({
    'Category': np.random.choice(['Complaint', 'Request', 'Feedback', 'Spam'], 100, p=[0.3, 0.4, 0.2, 0.1]),
    'Urgency': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.5, 0.3, 0.2]),
    'Date': pd.date_range(start='2024-01-01', periods=100)
})

# Filter (Requirement: "Filters")
filter_urgency = st.sidebar.multiselect("Filter by Urgency", options=['Low', 'Medium', 'High'], default=['Low', 'Medium', 'High'])
filtered_data = dummy_data[dummy_data['Urgency'].isin(filter_urgency)]

# Charts (Requirement: "Charts")
st.sidebar.subheader("Ticket Volume by Category")
fig_cat = px.bar(filtered_data['Category'].value_counts(), orientation='h')
st.sidebar.plotly_chart(fig_cat, use_container_width=True)

st.sidebar.subheader("Urgency Distribution")
fig_urg = px.pie(filtered_data, names='Urgency', hole=0.4)
st.sidebar.plotly_chart(fig_urg, use_container_width=True)


# --- MAIN AREA: REAL-TIME CLASSIFICATION ---
st.subheader("📝 Live Email Classification")
email_input = st.text_area("Paste Incoming Email Content Here:", height=150)

if st.button("Analyze Email"):
    if email_input:
        with st.spinner("Processing..."):
            # Get Predictions
            category = predict_category(email_input)
            urgency = predict_urgency(email_input)
            
            # Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("📂 Category Detected")
                st.markdown(f"### {category}")
                
            with col2:
                # Dynamic Color for Urgency
                if urgency == "High":
                    st.error("🚨 Urgency Level")
                elif urgency == "Medium":
                    st.warning("⚠️ Urgency Level")
                else:
                    st.success("✅ Urgency Level")
                st.markdown(f"### {urgency}")
                
            # Raw Data View
            st.markdown("---")
            st.caption("Processed Text sent to Model:")
            st.text(clean_email(email_input))
            
    else:
        st.warning("Please enter some text to analyze.")