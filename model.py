import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

def clean_text(text):
    """
    Cleans raw email text by lowering case and removing special characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-True\s]', '', text)  # Keep letters and numbers
    return text

def train_spam_classifier(data_path):
    """
    Loads data, preprocesses text, trains a Naive Bayes model, and saves it.
    """
    print("⏳ Loading dataset...")
    # Expects a CSV with 'text' and 'label' (or 'target') columns
    df = pd.read_csv(data_path)
    
    # Ensure correct columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    print("🧹 Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_train_split(
        df['cleaned_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )

    print("🏗️ Creating Machine Learning Pipeline...")
    # Combine TF-IDF Vectorization and Naive Bayes into a unified pipeline
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('classifier', MultinomialNB(alpha=0.1))
    ])

    print("🚀 Training the model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate the performance
    predictions = model_pipeline.predict(X_test)
    print("\n📊 Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save the pipeline object for deployment
    print("💾 Saving trained model to disk...")
    joblib.dump(model_pipeline, 'spam_classifier_model.pkl')
    print("✅ Model saved successfully as 'spam_classifier_model.pkl'!")

if __name__ == "__main__":
    # Replace with your actual dataset path when running locally
    # train_spam_classifier("data/spam_dataset.csv")
    pass
