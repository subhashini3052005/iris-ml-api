from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime
import joblib
import os
import json


iris = load_iris()

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression(max_iter=200)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


os.makedirs("ml/saved_model", exist_ok=True)

metadata = {
    "model_type":type(model).__name__,
    "version": "1.0.0",
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "features": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
}

with open("ml/saved_model/model_info.json","w")as f:
    json.dump(metadata, f, indent=4)

joblib.dump(model, "ml/saved_model/model.joblib")

print("Model saved successfully!")