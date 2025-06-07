# Phishing-Website-Detection-Using-Machine-Learning

Phishing attacks are one of the most common and dangerous threats in cybersecurity, often targeting unsuspecting users through fake websites designed to steal sensitive information such as usernames, passwords, and banking credentials. The goal of this project is to develop an intelligent machine learning system that can detect whether a website URL is legitimate or a phishing attempt, based purely on the structure and content of the URL.

This project automates phishing detection using key URL-based features and trains a machine learning model to classify suspicious sites in real-time. It is implemented in Google Colab and offers an easy-to-use web interface using Gradio, where users can input any URL and receive an immediate prediction.

Key Objectives
Extract meaningful features from website URLs.

Train a machine learning model (Random Forest) for classification.

Provide real-time predictions for new, unseen URLs.

Offer an interactive user interface via Gradio.

How It Works
Dataset
We use a labeled dataset of URLs (both phishing and legitimate), which includes various examples of how attackers try to deceive users.

Feature Engineering
Each URL is analyzed for specific patterns and characteristics often found in phishing websites, such as:

Use of IP address instead of domain name

Presence of suspicious characters like @, //, -, etc.

Abnormal length of the URL

Number of dots, slashes, digits, and symbols

Model Training
The extracted features are used to train a Random Forest Classifier, a robust ensemble machine learning model that handles binary classification effectively.

Evaluation
The model is evaluated using accuracy metrics and cross-validation to ensure it generalizes well to unseen data.

Real-Time Interface
A simple and intuitive Gradio UI is provided where users can enter any URL and instantly receive a classification result:

✅ Legitimate

❌ Phishing
Technologies Used
Python: Programming language for development

Pandas & NumPy: Data manipulation and analysis

Scikit-learn: Machine learning algorithms and evaluation

Gradio: Real-time user interface

Google Colab: Cloud-based environment for development and deployment

Results
The model achieves high accuracy on test data and performs well in real-time detection scenarios.

It offers an effective and fast way to flag phishing websites before users fall into the trap.

 Applications
Web browsers and plugins for phishing protection

Email clients to detect malicious URLs in messages

Cybersecurity education and training platforms

Enterprise security systems for safe browsing

