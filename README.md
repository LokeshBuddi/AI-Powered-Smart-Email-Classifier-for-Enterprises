# AI-Powered-Smart-Email-Classifier-for-Enterprises

# Customer Support Ticket & Email Classification System

This project is a Machine Learning pipeline designed to classify customer communications into four distinct categories: **Complaints**, **Requests**, **Spam**, and **Feedback**.

It processes text data from multiple sources, cleans it using NLP techniques, trains a Logistic Regression model, and generates statistical visualizations of the text data.

## 📌 Features

* **Multi-Source Data Integration**: Merges data from three different datasets (Customer Support Tickets, Spam Emails, and Amazon Reviews).
* **Text Preprocessing Pipeline**:
    * Lowercasing & Punctuation removal.
    * Regex cleaning (removing email addresses and URLs).
    * Stopword removal & Lemmatization (using NLTK).
* **Machine Learning**:
    * **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency).
    * **Model**: Logistic Regression.
* **Visualization**: Generates and saves boxplots for word and character count distributions.
* **Export**: Saves the cleaned, processed, and labeled dataset to a CSV file.

## 📂 Project Structure

```bash
├── main.py                     # The primary Python script
├── customer_support_tickets.csv # Dataset 1: Support tickets
├── mail_data.csv               # Dataset 2: Spam/Ham emails
├── amazon.csv                  # Dataset 3: Sentiment analysis data
├── final_processed_data.csv    # OUTPUT: The cleaned and merged dataset
├── distribution_plots.png      # OUTPUT: Visualization of text stats
├── zoomed_in_plots.png         # OUTPUT: Outlier-removed visualization
└── README.md                   # Project documentation

