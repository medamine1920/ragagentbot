import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from services.cassandra_connector import CassandraConnector

class ChatCategorizer:
    def __init__(self):
        self.model_path = "models/category_model.pkl"
        self.vectorizer_path = "models/vectorizer.pkl"
        self.session = CassandraConnector().get_session()
        self.load_model()

    def load_model(self):
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict_and_save(self):
        rows = self.session.execute("SELECT id, question, response FROM chat_history")
        for row in rows:
            text = f"{row.question or ''} {row.response or ''}".strip()
            if not text:
                continue
            vec = self.vectorizer.transform([text])
            category = self.model.predict(vec)[0]
            self.session.execute("""
                UPDATE chat_history SET category = %s WHERE id = %s
            """, (category, row.id))

    def retrain_model(self):
        rows = self.session.execute("SELECT question, response, category FROM chat_history WHERE category IS NOT NULL ALLOW FILTERING")
        data = [{"text": f"{r.question or ''} {r.response or ''}", "category": r.category} for r in rows]
        df = pd.DataFrame(data)
        X = df["text"]
        y = df["category"]

        self.vectorizer = TfidfVectorizer()
        X_vec = self.vectorizer.fit_transform(X)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X_vec, y)

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
