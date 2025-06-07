# ================================
# STEP 1: Install Required Packages
# ================================
!pip install gradio --quiet

# ================================
# STEP 2: Import Dependencies
# ================================
import pandas as pd
import numpy as np
import re
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import files

# ================================
# STEP 3: Upload and Load Dataset
# ================================
def upload_and_load_data():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    data = pd.read_csv(filename)
    if 'Label' not in data.columns or 'URL' not in data.columns:
        raise ValueError("Dataset must contain 'URL' and 'Label' columns.")
    data['label'] = data['Label'].apply(lambda x: 1 if x.lower() == 'bad' else 0)
    data.drop(columns=['Label'], inplace=True)
    return data

# ================================
# STEP 4: Feature Extraction Logic
# ================================
def extract_features(url):
    features = {
        'url_length': len(url),
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        'has_at': 1 if '@' in url else 0,
        'has_https': 1 if 'https' in url.lower() else 0,
        'count_dots': url.count('.'),
        'count_slash': url.count('/'),
        'count_hyphen': url.count('-'),
        'count_equal': url.count('='),
        'count_question': url.count('?'),
        'count_percent': url.count('%'),
        'count_digits': sum(c.isdigit() for c in url)
    }
    return list(features.values())

# ================================
# STEP 5: Prepare Features and Labels
# ================================
def prepare_dataset(df):
    X = df['URL'].apply(extract_features)
    X = pd.DataFrame(X.tolist(), columns=[
        'url_length', 'has_ip', 'has_at', 'has_https', 'count_dots',
        'count_slash', 'count_hyphen', 'count_equal', 'count_question',
        'count_percent', 'count_digits'
    ])
    y = df['label']
    return X, y

# ================================
# STEP 6: Train Machine Learning Model
# ================================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Accuracy on test set: {acc:.4f}")
    return clf

# ================================
# STEP 7: Build Gradio Interface
# ================================
def build_interface(model):
    def predict(url):
        features = extract_features(url)
        prediction = model.predict([features])[0]
        return "Phishing Website" if prediction == 1 else "Legitimate Website"

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Enter Website URL"),
        outputs=gr.Label(label="Prediction"),
        title="Phishing Website Detection System",
        description="This system uses a machine learning model trained on URL patterns to detect phishing websites."
    )
    interface.launch()

# ================================
# STEP 8: Main Execution Block
# ================================
def main():
    print("Upload a CSV file with 'URL' and 'Label' columns...")
    df = upload_and_load_data()
    X, y = prepare_dataset(df)
    model = train_model(X, y)
    build_interface(model)

# Run the pipeline
main()
# Step 1: Install required packages
!pip install python-whois requests gradio --quiet

# Step 2: Imports
import pandas as pd
import numpy as np
import re
import socket
import ssl
import whois
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import files

# Step 3: Upload dataset function
def upload_and_load_data():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(filename)
    if 'Label' not in df.columns or 'URL' not in df.columns:
        raise ValueError("Dataset must contain 'URL' and 'Label' columns.")
    df['label'] = df['Label'].apply(lambda x: 1 if x.lower() == 'bad' else 0)
    df.drop(columns=['Label'], inplace=True)
    return df

# Step 4: Enhanced feature extraction with WHOIS, SSL, Shortener checks
def enhanced_extract_features(url):
    features = {}

    features['url_length'] = len(url)
    features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    features['has_at'] = 1 if '@' in url else 0
    features['has_https'] = 1 if 'https' in url.lower() else 0
    features['count_dots'] = url.count('.')
    features['count_slash'] = url.count('/')
    features['count_hyphen'] = url.count('-')
    features['count_equal'] = url.count('=')
    features['count_question'] = url.count('?')
    features['count_percent'] = url.count('%')
    features['count_digits'] = sum(c.isdigit() for c in url)

    # Shortener detection
    shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly']
    features['is_shortened'] = int(any(s in url for s in shorteners))

    # WHOIS domain age
    def get_domain_age(domain):
        try:
            domain_info = whois.whois(domain)
            creation_date = domain_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            age_days = (pd.Timestamp.now() - pd.to_datetime(creation_date)).days
            return age_days if age_days > 0 else -1
        except Exception:
            return -1

    # SSL certificate validation
    def has_valid_ssl(domain):
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                s.settimeout(3.0)
                s.connect((domain, 443))
            return 1
        except Exception:
            return 0
          try:
        domain = re.findall(r"https?://([^/]+)", url)[0]
        features['domain_age'] = get_domain_age(domain)
        features['valid_ssl'] = has_valid_ssl(domain)
    except Exception:
        features['domain_age'] = -1
        features['valid_ssl'] = 0

    return list(features.values())

# Step 5: Prepare dataset features and labels
def prepare_dataset(df):
    X = df['URL'].apply(enhanced_extract_features)
    feature_names = [
        'url_length', 'has_ip', 'has_at', 'has_https', 'count_dots',
        'count_slash', 'count_hyphen', 'count_equal', 'count_question',
        'count_percent', 'count_digits', 'is_shortened', 'domain_age', 'valid_ssl'
    ]
    X = pd.DataFrame(X.tolist(), columns=feature_names)
    y = df['label']
    return X, y

# Step 6: Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Test accuracy: {acc:.4f}")
    return clf

# Step 7: Gradio interface for live prediction
def build_interface(model):
    feature_names = [
        'url_length', 'has_ip', 'has_at', 'has_https', 'count_dots',
        'count_slash', 'count_hyphen', 'count_equal', 'count_question',
        'count_percent', 'count_digits', 'is_shortened', 'domain_age', 'valid_ssl'
    ]

    def predict(url):
        features = enhanced_extract_features(url)
        prediction = model.predict([features])[0]
        return "Phishing Website" if prediction == 1 else "Legitimate Website"

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Enter Website URL"),
        outputs=gr.Label(label="Prediction"),
        title="Enhanced Phishing Website Detection",
        description="Enter a URL to check if it is a phishing website or legitimate."
    )
    interface.launch()

# Step 8: Main function to run pipeline
def main():
    print("Please upload your dataset CSV file with 'URL' and 'Label' columns.")
    df = upload_and_load_data()
    X, y = prepare_dataset(df)
    model = train_model(X, y)
    build_interface(model)

# Run
main()
