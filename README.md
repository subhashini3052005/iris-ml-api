# Iris ML API

## Project Overview

This project builds a REST API that serves a machine learning model for Iris flower classification. The goal is to take flower measurements through an API request and return the predicted Iris species.

## Dataset

The project uses the Iris flower dataset. It contains measurements of Iris flowers and their corresponding species.

### Input Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

  ## Target

  The model predicts one of three Iris species:

  -Setosa
  -Versicolor
  -Virginica

  ## Machine Learning Problem

  This is a supervised learning classification problem because the training data contains both input features and the correct species labels.

  ## Model

  The project will use Logistic Regression as the machine learning classification model.

  ## API Contract

  The '/predict' endpoint will accept four numerical measurements of an Iris flower :sepal length, sepal width, petal length and petal width. The API validate these inputs before passing them to the trained machine learning model.If the input is valid, the model will predict the Iris flower species as Setosa,Versicolor, or virginica. The API will return the predicted species in a JSON response. If the input is invalid, the API will return an appropriate error response instead of making prediction.

  ## Request Flow

  Client Request
  -> Input Validation
  -> ML Model
  -> Prediction
  -> JSON Response

  ## MVP Scope

  The minimum viable product will include:

  - One '/predict' endpoint
  - Iris classification model
  - Input validation
  - JSON prediction response
