import warnings

# ======================================
#  Evaluate the models on the test set
# ======================================

warnings.simplefilter("ignore")


def rmse(errors):
    """Return the root mean square error."""
    return np.sqrt(np.mean(errors ** 2))


def process_test_set(data):
    """Performs the necessary pre-processing steps in the pipeline to the test set"""
    # Add time-based features and the speed columns
    data = pf.add_features(data)

    # Add the region column
    data = pf.add_region(data, config.REGION_BOUNDS)

    # Apply cube-root transformation
    for var in config.CBRT_TRANSFORM:
        data[var] = pf.cbrt_transform(data, var)

    # Scale the numerical data
    scaled = data[config.NUM_VARS].copy()
    scaled.iloc[:, :] = pf.scale_features(scaled, config.SCALER_PATH)

    # One-hot encode all the categorical variables
    categoricals = []
    for var in config.CAT_VARS:
        categoricals.append(pf.encode_categorical(data, var))

    # Final design matrix for training
    return pf.concat_dfs(scaled, categoricals)


# ============================================================================

def main():
    try:
        arg = sys.argv[1]
    except IndexError:
        print("Please specify a model to test!")
        return
    if arg not in [config.LR_ARG, config.LR_SPEED_ARG, config.NN_ARG]:
        print("Unknown argument ", arg)
        return

    # Load the data
    data = pf.load_data(config.PATH_TO_DATASET)
    # Add the speed variable to data
    data[config.AUX_TARGET] = pf.speed(data)
    # Divide the data into training and test sets, with target TARGET
    X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)
    # Save the distance column before scaling (for converting speed)
    distance = X_test['distance']

    X_test = process_test_set(X_test)

    if arg == config.LR_ARG:
        preds = pf.predict(X_test, config.LINEAR_REG_MODEL_PATH) ** 3
    if arg == config.NN_ARG:
        preds = pf.predict(X_test, config.NEURAL_NET_MODEL_PATH)
        pred_n = preds.shape[0]
        preds = preds.reshape((pred_n,)) ** 3
    if arg == config.LR_SPEED_ARG:
        preds = pf.predict(X_test, config.LINEAR_REG_SPEED_MODEL_PATH)
        preds = (1 / preds) * distance * 3600

    constant_model = np.mean(y_train)
    rmse_constant = rmse(y_test - constant_model)
    test_rmse = rmse(y_test - preds)

    print("-----------------------------------------|-----------------------")
    print("Baseline (predict average duration) RMSE |", rmse_constant)
    print("-----------------------------------------|-----------------------")
    print("               Test RMSE                 |", test_rmse)
    print("-----------------------------------------|-----------------------")


if __name__ == "__main__":
    import numpy as np
    import sys
    import config
    import preprocessing_functions as pf

    main()
