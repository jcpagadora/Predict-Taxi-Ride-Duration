import numpy as np
import pandas as pd

from preprocessing_functions import Pipeline
import config

pipeline = Pipeline(features=config.FEATURES,
                    y=config.TARGET,
                    time_pickup=config.TIME_PICKUP,
                    location_pickup=config.LOCATION_PICKUP,
                    var_to_cbrt=config.CBRT_TRANSFORM,
                    cat_vars=config.CAT_VARS,
                    num_vars=config.NUM_VARS)

if __name__ == '__main__':
    data = pd.read_csv(config.PATH_TO_DATASET)

    pipeline.fit(data)
    pipeline.eval_model()
    predictions = pipeline.predict(data)
    print(predictions[:10])
