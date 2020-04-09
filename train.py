import config
import preprocessing_functions as pf
import warnings

# =================================================================================================
#  Load the dataset, perform pre-processing, feature engineering & selection, and train the models
# =================================================================================================

warnings.simplefilter("ignore")

# Load the data
data = pf.load_data(config.PATH_TO_DATASET)

# Add the speed variable to data
data[config.AUX_TARGET] = pf.speed(data)

# Divide the data into training and test sets, with target TARGET
X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)

# Add time-based features and the speed columns
X_train = pf.add_features(X_train)

# Add the region column
X_train = pf.add_region(X_train, config.REGION_BOUNDS)

# Apply cube-root transformation
for var in config.CBRT_TRANSFORM:
    X_train[var] = pf.cbrt_transform(X_train, var)

# Train standard scaler on numerical variables only
scaled = X_train[config.NUM_VARS].copy()
scaler = pf.train_scaler(scaled, config.SCALER_PATH)

# Scale the numerical data
scaled.iloc[:,:] = pf.scale_features(scaled, config.SCALER_PATH)

# One-hot encode all the categorical variables
categoricals = []
for var in config.CAT_VARS:
    categoricals.append(pf.encode_categorical(X_train, var))

# Final design matrix for training
X_train = pf.concat_dfs(scaled, categoricals)

# Assert we have the desired features
assert X_train.columns.tolist() == config.FEATURES

# Train the default linear regression model
pf.train_linreg_model(X_train, y_train, config.LINEAR_REG_MODEL_PATH)

# Train the linear regression model via speed
pf.train_linreg_model(X_train, y_train, config.LINEAR_REG_SPEED_MODEL_PATH)

# Train the neural network
pf.train_nn_model(X_train, y_train, config.NET_ARCHITECTURE_AND_PARAMETERS, config.NEURAL_NET_MODEL_PATH)

print("Training Finished Successfully! You can now run eval.py.")
