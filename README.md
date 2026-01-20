# 📧 AI-Powered Smart Email Classifier for Enterprises

An intelligent Machine Learning pipeline designed to automate customer support workflows by classifying incoming communications into four distinct categories: **Complaints**, **Requests**, **Spam**, and **Feedback**.

This solution helps enterprises prioritize critical issues, reduce response times, and filter out noise automatically.

## 🚀 Live Demo
**[Click here to view the Live Dashboard](https://huggingface.co/spaces/lokeshbuddhi/email-classifier-dashboard)** *(Hosted on Hugging Face Spaces)*

---

## 📌 Features

### 1. Multi-Source Data Integration
Merges diverse datasets to create a robust training ground:
* **Customer Support Tickets:** Real-world support queries.
* **Spam Emails:** Dataset for detecting junk mail.
* **Amazon Reviews:** Sentiment data for feedback analysis.

### 2. Advanced NLP Preprocessing Pipeline
* **Cleaning:** Lowercasing, punctuation removal, and Regex cleaning (removing email addresses and URLs).
* **Normalization:** Stopword removal and Lemmatization using NLTK.
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into machine-readable numbers.

### 3. Machine Learning Model
* **Algorithm:** Logistic Regression (optimized for text classification).
* **Performance:** Classifies text into `Complaint`, `Request`, `Spam`, or `Feedback`.
* **Urgency Detection:** Simultaneously predicts urgency levels (High/Medium/Low) using a secondary model layer.

### 4. Visualization & Reporting
* Generates statistical boxplots for word and character count distributions.
* Saves cleaned and labeled datasets for further analysis.
* **Interactive Dashboard:** A Streamlit-based UI for real-time testing.

---

## 📂 Project Structure

```
├── app.py                      # The main Streamlit dashboard application
├── main.py                     # Script for data processing and model training
├── requirements.txt            # List of dependencies
├── saved_models/               # Trained models (TF-IDF vectorizer & Logistic Regression)
│   ├── vectorizer.pkl
│   └── logistic_model.pkl
├── customer_support_tickets.csv # Source Dataset 1
├── mail_data.csv               # Source Dataset 2
├── amazon.csv                  # Source Dataset 3
├── final_processed_data.csv    # OUTPUT: Cleaned and merged dataset
├── distribution_plots.png      # OUTPUT: Visualization of text stats
└── README.md                   # Project documentation 
```
---

## 🛠️ Installation & Usage

Prerequisites
Python 3.8+



### 1. Clone the Repository

git clone [https://github.com/lokeshbuddi/AI-Powered-Smart-Email-Classifier-for-Enterprises.git](https://github.com/lokeshbuddi/AI-Powered-Smart-Email-Classifier-for-Enterprises.git)
cd AI-Powered-Smart-Email-Classifier-for-Enterprises

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Run the Dashboard

streamlit run app.py
The app will open in your browser at http://localhost:8501.

## 📊 How It Works

### Input: 
The user pastes an email into the text box on the dashboard.

### Process: 
The system cleans the text and runs it through the pre-trained Logistic Regression and DistilBERT models.

### Output: 
The dashboard displays:

Category: (e.g., "Complaint")

Urgency: (e.g., "High Priority")

Confidence Score: Probability percentage.

## 👨‍💻 Author

### Lokesh Buddi
