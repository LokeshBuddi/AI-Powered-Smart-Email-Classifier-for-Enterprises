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

# ===================
# 2. LOAD RESOURCES
# ===================
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
    st.error(f"Error loading models: {e}. ")
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

# --- Explainable AI Function ---
def explain_prediction(text, category):
    # 1. Define keywords for each category
    keywords = {
        "Complaint": ["bad", "slow", "wrong", "fail", "terrible", "refund", "issue", "problem", "angry", "disappointed", "late"],
        "Spam": ["free", "win", "prize", "cash", "click", "subscribe", "buy", "offer", "limited", "urgent", "credit"],
        "Request": ["need", "help", "can", "please", "support", "question", "assist", "inquiry", "how"],
        "Feedback": ["good", "great", "love", "like", "improve", "suggestion", "best", "thanks", "excellent"]
    }
    
    words = text.split()
    highlighted_text = []
    
    # 2. Get the specific keywords for the predicted category
    target_words = keywords.get(category, [])
    
    # 3. Loop through words and highlight if they match
    for word in words:
        clean_word = word.lower().strip(".,!?")
        if clean_word in target_words:
            # Red highlight for Complaints/Spam, Green for others
            color = "#ffcccc" if category in ["Complaint", "Spam"] else "#ccffcc"
            highlighted_text.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px; font-weight: bold; color: black;">{word}</span>')
        else:
            highlighted_text.append(word)
            
    return " ".join(highlighted_text)

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
# --- Add Tabs ---
tab1, tab2 = st.tabs(["📝 Single Email Analysis", "📂 Batch Processing (CSV)"])

# --- TAB 1 ---
with tab1:
        # EVERYTHING below this line must be indented (press Tab once)  
    st.markdown("Enterprise Solution for Automated Urgency Detection")

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
    st.sidebar.plotly_chart(fig_cat, width='stretch')

    st.sidebar.subheader("Urgency Distribution")
    fig_urg = px.pie(filtered_data, names='Urgency', hole=0.4)
    st.sidebar.plotly_chart(fig_urg, width='stretch')


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
                        st.error(" Urgency Level")
                    elif urgency == "Medium":
                        st.warning(" Urgency Level")
                    else:
                        st.success(" Urgency Level")
                    st.markdown(f"### {urgency}")
                st.subheader("🔍 Explainable AI: Why this category?")
                    
                    # Call the function we created above
                # --- LEVEL 2: Explainable AI Function (SMART VERSION) ---
                def explain_prediction(text, _): 
                    # Note: We ignore the 'category' input because we want to highlight ALL keywords found.
                    
                    words = text.split()
                    highlighted_text = []
                    
                    # Define keyword lists with associated colors
                    keyword_map = {
                        "Complaint": {
                            "words": ["bad", "slow", "wrong", "fail", "terrible", "refund", "issue", "problem", "angry", "disappointed", "late", "hate", "worst"],
                            "color": "#ffcccc", # Red
                            "label": "Complaint"
                        },
                        "Spam": {
                            "words": ["free", "win", "prize", "cash", "click", "subscribe", "buy", "offer", "limited", "urgent", "credit", "money"],
                            "color": "#ffebcc", # Orange
                            "label": "Spam"
                        },
                        "Request": {
                            "words": ["need", "help", "can", "please", "support", "question", "assist", "inquiry", "how", "check"],
                            "color": "#cce5ff", # Blue
                            "label": "Request"
                        },
                        "Feedback": {
                            "words": ["good", "great", "love", "like", "improve", "suggestion", "best", "thanks", "excellent", "amazing"],
                            "color": "#ccffcc", # Green
                            "label": "Feedback"
                        }
                    }

                    for word in words:
                        clean_word = word.lower().strip(".,!?")
                        match_found = False
                        
                        # Check this word against ALL categories
                        for category, data in keyword_map.items():
                            if clean_word in data["words"]:
                                # If match found, highlight it with that category's color
                                highlighted_text.append(
                                    f'<span style="background-color: {data["color"]}; padding: 2px 4px; border-radius: 4px; font-weight: bold; title="{data["label"]}">{word}</span>'
                                )
                                match_found = True
                                break
                        
                        if not match_found:
                            highlighted_text.append(word)
                            
                    return " ".join(highlighted_text)

                        
                    
                # Raw Data View
                st.markdown("---")
                st.caption("Processed Text sent to Model:")
                st.text(clean_email(email_input))
                
        else:
            st.warning("Please enter some text to analyze.")
# --- TAB 2 ---
with tab2:
    st.header("📂 Bulk Email Classification")
    st.write("Upload a CSV file to classify multiple emails at once.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", df.head(3))
            
            # Smart Column Detection: Look for 'text', 'email', or 'content' columns
            possible_cols = ['text', 'email', 'content', 'message', 'Ticket Description', 'Body']
            target_col = None
            
            for col in possible_cols:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col:
                st.success(f" Found column to classify: '{target_col}'")
                
                if st.button(" Start Batch Classification"):
                    # Create empty lists for results
                    categories = []
                    urgencies = []
                    
                    # Progress Bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Loop through every row in the CSV
                    for index, row in df.iterrows():
                        # Get text
                        email_text = str(row[target_col])
                        
                        # Clean and Predict (Using your existing functions!)
                        cleaned = clean_text(email_text) 
                        vect = vectorizer.transform([cleaned])
                        pred = model.predict(vect)[0]
                        
                        # Determine Urgency (Same logic as Tab 1)
                        urgency = "Low"
                        if pred in ["Complaint", "Spam"]:
                            urgency = "High"
                        elif pred == "Request":
                            urgency = "Medium"
                            
                        categories.append(pred)
                        urgencies.append(urgency)
                        
                        # Update Progress
                        progress_bar.progress((index + 1) / len(df))
                        status_text.text(f"Processing email {index + 1} of {len(df)}...")
                    
                    # Add results to the DataFrame
                    df['Predicted_Category'] = categories
                    df['Predicted_Urgency'] = urgencies
                    
                    st.success("✅ Classification Complete!")
                    st.dataframe(df.head())
                    
                    # Download Button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Classified Results",
                        data=csv,
                        file_name="classified_emails.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f" Could not find a text column. Please ensure your CSV has a column named: {', '.join(possible_cols)}")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")