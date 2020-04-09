import numpy as np
import pandas as pd
import sqlite3
import datetime
import warnings

# ===============================================================================
# Run this file to load the data set and perform the necessary cleaning
# before starting the pre-processing (feature engineering and selection)
# ===============================================================================

warnings.simplefilter("ignore")

# Connect to the taxi database
conn = sqlite3.connect('taxi.db')

# We will select all taxi rides within the following boundaries which approximately
# contains all of Manhattan
lon_bounds = [-74.03, -73.75]
lat_bounds = [40.6, 40.88]


def cons_bound_str(field, bounds):
    """Helper function that constructs the string used for the query in selecting datapoints
       where the given field is within the given boundaries"""
    return field + " >= " + str(bounds[0]) + " AND " + field + " <= " + str(bounds[1])

pickup_bound_lon = cons_bound_str("pickup_lon", lon_bounds)
pickup_bound_lat = cons_bound_str("pickup_lat", lat_bounds)
dropoff_bound_lon = cons_bound_str("dropoff_lon", lon_bounds)
dropoff_bound_lat = cons_bound_str("dropoff_lat", lat_bounds)

query = "SELECT * FROM taxi WHERE " + pickup_bound_lon + " AND " \
        + pickup_bound_lat + " AND " + dropoff_bound_lon + " AND " + dropoff_bound_lat

taxi = pd.read_sql(query, conn)

# Now let's select the taxi rides
polygon = pd.read_csv('manhattan.csv')

def in_manhattan(x, y):
    """ Checks whether the given point (x,y) is in Manhattan.
    >>> in_manhattan(-74.013513, 40.712317)
    True
    >>> in_manhattan(-73.941, 40.816)
    True
    >>> in_manhattan(-73.9457, 40.7555)
    False
    >>> in_manhattan(-74.034128, 40.730788)
    False
    """
    num_corners = len(polygon)
    poly_x = polygon['lon']
    poly_y = polygon['lat']
    j = num_corners - 1
    odd_nodes = False
    for i in range(num_corners):
        if (poly_y[i] < y and poly_y[j] >= y) or (poly_y[j] < y and poly_y[i] >= y):
            if poly_x[i] + (y - poly_y[i]) / (poly_y[j] - poly_y[i]) * (poly_x[j] - poly_x[i]) < x:
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes

pickup_in_manhattan = np.array([in_manhattan(x, y) for x,y in taxi[['pickup_lon', 'pickup_lat']].values])
dropoff_in_manhattan = np.array([in_manhattan(x, y) for x,y in taxi[['dropoff_lon', 'dropoff_lat']].values])
manhattan_taxi = taxi[pickup_in_manhattan & dropoff_in_manhattan]

# Finally, select valid taxi rides: positive passenger count, positive distance, duration
# of at least 1 minute and at most 1 hour, and an average speed of at most 100 miles per hour
manhattan_taxi = manhattan_taxi.query("passengers > 0 and distance > 0 and duration >= 60 and \
                                            duration <= 60*60 and distance / duration <= 100/3600")

# Now convert pickup_datetime to date object
manhattan_taxi.loc[:, 'date'] = manhattan_taxi['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))

# Recall from our research (see the Predicting Manhattan Taxi Ride Duration notebook), we should remove some days
atypical = [1, 2, 3, 18, 23, 24, 25, 26]
typical_dates = [datetime.date(2016, 1, n) for n in range(1, 32) if n not in atypical]
manhattan_taxi_final = manhattan_taxi[manhattan_taxi['date'].isin(typical_dates)]

#print(manhattan_taxi.shape)

# Finally, save our cleaned data set
manhattan_taxi_final.to_csv('manhattan_taxi.csv', index=False)

# =================================================================================

if __name__ == "__main__":
    import doctest

    doctest.testmod()
