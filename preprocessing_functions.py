import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense


class Pipeline:

    def __init__(self, features, y, time_pickup, location_pickup, var_to_cbrt,
                 cat_vars, num_vars, test_size=0.2, random_state=0):
        # Initialize data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.features = features
        self.y = y

        # Initialize features to engineer
        self.time_pickup = time_pickup
        self.location_pickup = location_pickup
        self.var_to_cbrt = var_to_cbrt
        self.cat_vars = cat_vars
        self.num_vars = num_vars

        # Region bounds, to be learned from training
        self.bounds = []

        # Initialize scalers and models
        self.scaler = StandardScaler()
        self.model = LinearRegression()

        self.test_size = test_size
        self.random_state = random_state

    def add_time_vars(self, df):
        pickup_time = pd.to_datetime(df[self.time_pickup])
        df.loc[:, 'hour'] = pickup_time.dt.hour
        df.loc[:, 'day'] = pickup_time.dt.weekday
        return df

    def cbrt_transform(self, df):
        df.loc[:, "cbrt_" + self.var_to_cbrt] = np.cbrt(df[self.var_to_cbrt])
        return df

    def calculate_regions(self):
        first_pc = calc_first_pc(self.X_train)
        self.bounds = pd.qcut(first_pc, 3, retbins=True)[1]

    def add_regions(self, df):
        first_pc = calc_first_pc(df)
        bounds = self.bounds
        df.loc[:, 'region'] = np.zeros(len(df))
        df.loc[:, 'region'] = np.where((bounds[0] < first_pc) & (first_pc <= bounds[1]), 0, df['region'])
        df.loc[:, 'region'] = np.where((bounds[1] < first_pc) & (first_pc <= bounds[2]), 1, df['region'])
        df.loc[:, 'region'] = np.where((bounds[2] < first_pc) & (first_pc <= bounds[3]), 2, df['region'])
        return df

    def encode_categorical(self, df):
        """Perform one-hot encoding for the categorical variable var."""
        categoricals = []
        for var in self.cat_vars:
            categoricals.append(pd.get_dummies(df[var], prefix=var, drop_first=True))
        return categoricals

    def fit(self, data):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, data[self.y],
            test_size=self.test_size,
            random_state=self.random_state)

        self.X_train = self.add_time_vars(self.X_train)
        self.X_test = self.add_time_vars(self.X_test)

        self.X_train = self.cbrt_transform(self.X_train)
        self.X_test = self.cbrt_transform(self.X_test)

        self.calculate_regions()
        self.X_train = self.add_regions(self.X_train)
        self.X_test = self.add_regions(self.X_test)

        self.scaler.fit(self.X_train[self.num_vars])

        scaled_train = self.X_train[self.num_vars].copy()
        scaled_test = self.X_test[self.num_vars].copy()
        scaled_train.iloc[:, :] = self.scaler.transform(scaled_train)
        scaled_test.iloc[:, :] = self.scaler.transform(scaled_test)

        categoricals_train = self.encode_categorical(self.X_train)
        categoricals_test = self.encode_categorical(self.X_test)

        self.X_train = concat_dfs(scaled_train, categoricals_train)
        self.X_test = concat_dfs(scaled_test, categoricals_test)

        self.model.fit(self.X_train, np.cbrt(self.y_train))

        return self

    def transform(self, data):
        data = self.add_time_vars(data)
        data = self.cbrt_transform(data)
        data = self.add_regions(data)
        scaled = data[self.num_vars].copy()
        scaled.iloc[:, :] = self.scaler.transform(scaled)
        categoricals = self.encode_categorical(data)
        data = concat_dfs(scaled, categoricals)
        return data

    def predict(self, data):
        data = self.transform(data)
        predictions = self.model.predict(data) ** 3
        return predictions

    def eval_model(self):
        preds = self.model.predict(self.X_train) ** 3
        # Print training RMSE
        rmse = np.sqrt(np.mean((self.y_train - preds) ** 2))
        print("Training RMSE: ", rmse)

        preds = self.model.predict(self.X_test) ** 3
        # Print training RMSE
        rmse = np.sqrt(np.mean((self.y_test - preds) ** 2))
        print("Test-set RMSE: ", rmse)


# ===================================================
#  Individual pre-processing and training functions
# ===================================================

def calc_first_pc(df):
    D = df[['pickup_lon', 'pickup_lat']]
    pca_n = len(df)
    pca_means = D.apply(np.mean, axis=0)
    X = (D - pca_means) / np.sqrt(pca_n)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return (X @ vt.T)[0]


def load_data(df_path):
    """Load the data from df_path as a pandas dataframe"""
    return pd.read_csv(df_path)


def divide_train_test(df, target):
    """Divides the dataframe into a training and test set, with specified target variable"""
    X_train, X_test, y_train, y_test = train_test_split(df, df[target], test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


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


def cbrt_transform(data, var=None):
    """Perform a cube-root transformation on the given variable."""
    if not var:
        return np.cbrt(data)
    return np.cbrt(data[var])


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
