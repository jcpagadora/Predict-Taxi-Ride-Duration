# Predict-Taxi-Ride-Duration

This is a simple data science project that solves the following regression problem: predict the duration (in seconds) of a taxi trip in Manhattan given the pickup and dropoff times and coordinates, in addition to the distance of the trip. All research (exploratory data analysis, feature engineering, and feature selection) is done in the notebook
`Predicting Manhattan Taxi Ride Duration.ipynb`.


## Directions for default usage
1. Make sure to have the following packages installed:
  -`numpy`
  -`pandas`
  -`sklearn`
  -`keras`
  -`joblib`
  -`warnings`
2. Run `load_and_save_data.py`
3. Run `train.py`
4. Run `eval.py <model>`, where `<model>` is any of `lr`, `lr_speed`, or `nn`. `lr` is the model for a regular linear regression model, `lr_speed` is a linear regression model that predicts speed first before converting to duration, and `nn` uses a neural network to predict the duration.

Please see `config.py` to make any desired changes to the process.
