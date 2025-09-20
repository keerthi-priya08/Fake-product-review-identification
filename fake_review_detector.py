import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
df = pd.read_csv("large_dataset.csv")

# Step 2: Preprocess text (remove punctuation and lower case)
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['TEXT'] = df['TEXT'].apply(preprocess)

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['TEXT'], df['label'], test_size=0.2, random_state=42)

# Step 4: Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Real-time review prediction
while True:
    review = input("\nEnter a review to check (or type 'exit' to quit):\n> ")
    if review.lower() == 'exit':
        break
    review_processed = preprocess(review)
    review_vector = vectorizer.transform([review_processed])
    prediction = model.predict(review_vector)
    print("Prediction:", "Fake Review ❌" if prediction[0] == 1 else "Genuine Review ✅")
