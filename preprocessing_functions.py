import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense


# ===================================================
#  Individual pre-processing and training functions
# ===================================================

def load_data(df_path):
    """Load the data from df_path as a pandas dataframe"""
    return pd.read_csv(df_path)


def divide_train_test(df, target):
    """Divides the dataframe into a training and test set, with specified target variable"""
    X_train, X_test, y_train, y_test = train_test_split(df, df[target], test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def speed(df):
    """Takes in the dataframe and returns the average speed in mph"""
    return df['distance'] / df['duration'] * 60 * 60


def add_features(df):
    """Adds the hour, day, weekend, time_of_day categorical variables"""
    pickup_time = pd.to_datetime(df['pickup_datetime'])
    df.loc[:, 'hour'] = pickup_time.dt.hour
    df.loc[:, 'day'] = pickup_time.dt.weekday
    df.loc[:, 'weekend'] = (pickup_time.dt.weekday >= 5).astype(int)
    df.loc[:, 'time_of_day'] = np.digitize(pickup_time.dt.hour, [0, 6, 18])
    return df


def add_region(df, bounds):
    """Adds a region column for the data. This is computed using the first principal component
     of the pickup coordinates (see the notebook for details) and dividing Manhattan into
     separate regions using these values in the training set."""

    # Calculate the first principal component
    D = df[['pickup_lon', 'pickup_lat']]
    pca_n = len(df)
    pca_means = D.apply(np.mean, axis=0)
    X = (D - pca_means) / np.sqrt(pca_n)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    first_pc = (X @ vt.T)[0]

    # Number of regions. Assuming large amount of data, any overlap here should be fine.
    num_regions = len(bounds) - 1
    df['region'] = np.repeat('?', len(df))
    for i in range(num_regions):
        if i == 0:
            df.loc[:, 'region'] = np.where((df['region'] == '?') & (first_pc <= bounds[i + 1]), i, df['region'])
        elif i == num_regions - 1:
            df.loc[:, 'region'] = np.where((df['region'] == '?') & (bounds[i] <= first_pc), i, df['region'])
        else:
            df.loc[:, 'region'] = np.where((df['region'] == '?') & (bounds[i] <= first_pc)
                                           & (first_pc <= bounds[i + 1]), i, df['region'])
    assert np.sum(df['region'] == '?') == 0  # Mistake in assigning region to a taxi ride location
    return df


def cbrt_transform(df, var):
    """Perform a cube-root transformation on the given variable."""
    return np.cbrt(df[var])


def concat_dfs(df1, lst_dfs):
    """Concatenate dataframe 1 with a list of dataframes."""
    return pd.concat([df1] + lst_dfs, axis=1)


def encode_categorical(df, var):
    """Perform one-hot encoding for the categorical variable var."""
    return pd.get_dummies(df[var], prefix=var, drop_first=True)


def train_scaler(df, output_path):
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)  # Save the scaler
    return scaler


def scale_features(df, output_path):
    scaler = joblib.load(output_path)
    return scaler.transform(df)


def train_linreg_model(df, target, output_path):
    """Train a simple linear regression model and save it."""
    model = LinearRegression()
    model.fit(df, target)
    joblib.dump(model, output_path)  # Save the model


def train_nn_model(df, target, architecture, output_path):
    """Train a neural network and save it."""
    net = Sequential()
    for i in range(architecture['n_hidden_layers']):
        d = architecture[i]['dim']
        act_fn = architecture[i]['act_fn']
        if i == 0:
            net.add(Dense(d, input_dim=architecture['input_dim'], activation=act_fn))
        else:
            net.add(Dense(d, activation=act_fn))
    net.add(Dense(1))
    optimizer, loss, metrics = architecture['optimizer'], architecture['loss'], architecture['metrics']
    net.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    net.fit(df, target, epochs=architecture['epochs'], batch_size=architecture['batch_size'])
    joblib.dump(net, output_path)


def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)
