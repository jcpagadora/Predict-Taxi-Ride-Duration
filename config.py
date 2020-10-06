PATH_TO_DATASET = "manhattan_taxi.csv"
SCALER_PATH = 'scaler.pkl'
LINEAR_REG_MODEL_PATH = 'linear_regression.pkl'
LINEAR_REG_SPEED_MODEL_PATH = 'linear_regression_speed.pkl'
NEURAL_NET_MODEL_PATH = 'neural_net.pkl'

TARGET = 'duration'

# For categorical variable 'region,' using PCA
REGION_BOUNDS = [-2.65160160e-04, -4.79023218e-05, 4.64432561e-05, 5.70759348e-04]

NUM_VARS = ['pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat', 'cbrt_distance']
TIME_PICKUP = 'pickup_datetime'
LOCATION_PICKUP = ['pickup_lon', 'pickup_lat']
CAT_VARS = ['hour', 'day', 'region']

FEATURES = ['pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat', 'cbrt_distance',
            'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7',
            'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13',
            'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
            'hour_20', 'hour_21', 'hour_22', 'hour_23', 'day_1', 'day_2', 'day_3',
            'day_4', 'day_5', 'day_6', 'region_1.0', 'region_2.0']

CBRT_TRANSFORM = 'distance'

NET_ARCHITECTURE_AND_PARAMETERS = {
    'n_hidden_layers': 3,
    'input_dim': len(FEATURES),
    0: {'dim': 64, 'act_fn': 'relu'},
    1: {'dim': 32, 'act_fn': 'sigmoid'},
    2: {'dim': 16, 'act_fn': 'relu'},
    'optimizer': 'adam',
    'loss': 'mean_squared_error',
    'metrics': ['mape'],
    'epochs': 7,
    'batch_size': 10
}

LR_ARG = 'lr'
LR_SPEED_ARG = 'lr_speed'
NN_ARG = 'nn'
