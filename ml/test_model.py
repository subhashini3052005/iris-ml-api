import joblib


model = joblib.load("ml/saved_model/model.joblib")


sample = [[5.1, 3.5, 1.4, 0.2]]


prediction = model.predict(sample)

print("Prediction:", prediction)