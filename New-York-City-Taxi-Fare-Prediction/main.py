from helpers import *

import mpu

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def calculate_distance_driven(row):
    distance = mpu.haversine_distance((row["pickup_latitude"], row["pickup_longitude"]),
                                      (row["dropoff_latitude"], row["dropoff_longitude"]))
    return round(distance, 1)


def calculate_distance_from_nearest_airport(row):
    airports = [(40.64, -73.77), (40.77, -73.87), (40.69, -74.17)]
    minimum_distance = None
    for airport in airports:
        pickup_distance = round(mpu.haversine_distance((row["pickup_latitude"], row["pickup_longitude"]),
                                                       (airport[0], airport[1])), 2)
        dropoff_distance = round(mpu.haversine_distance((row["dropoff_latitude"], row["dropoff_longitude"]),
                                                        (airport[0], airport[1])), 2)
        closest_distance = min(pickup_distance, dropoff_distance)
        if not minimum_distance or closest_distance < minimum_distance:
            minimum_distance = closest_distance
    return minimum_distance


def bin_distance(df):
    bin_num = 8
    labels = [i for i in range(bin_num)]
    df['binned_distance_driven'] = pd.qcut(df['distance_driven'], bin_num, labels=labels)
    df.drop("distance_driven", axis=1, inplace=True)
    df[['binned_distance_driven']] = df[['binned_distance_driven']].astype(u'int8')
    return df


def bin_distance_from_airport(df):
    df['near_airport'] = pd.cut(df['distance_from_airport'], [0, 1, 100000], labels=[1, 0], right=True)
    df.drop("distance_from_airport", axis=1, inplace=True)
    df[['near_airport']] = df[['near_airport']].astype(bool)
    return df


def one_hot_encode_pickup_time(df):
    df['pickup_datetime'] = pd.cut(df.pickup_datetime.dt.hour, [i for i in range(0, 25, 3)], right=False)
    encoding = pd.get_dummies(df['pickup_datetime'], prefix='time_bin')
    df = pd.concat([df, encoding], axis=1)
    df.drop("pickup_datetime", axis=1, inplace=True)
    return df


def save_raw_as_feather():
    train = pd.read_csv("./input/train.csv", parse_dates=['pickup_datetime'])
    test = pd.read_csv("./input/test.csv", parse_dates=['pickup_datetime'])
    df = pd.concat([train, test], sort=False)
    df.reset_index().to_feather('raw_feather.feather')


def save_clean_as_feather(df):
    df.reset_index().to_feather('clean_feather.feather')


def load_raw_feather():
    df = pd.read_feather('raw_feather.feather')
    df = df.set_index(keys="key")
    df.drop("index", inplace=True, axis=1)
    return df


def load_clean_feather():
    df = pd.read_feather('clean_feather.feather')
    df = df.set_index(keys="key")
    df.drop("index", inplace=True, axis=1)
    return df


# save_raw_as_feather()
df = load_raw_feather()

print(df.head())

# df.dropna()     #drop all rows that have any NaN values

df[['passenger_count']] = df[['passenger_count']].astype(u'int8')

df["distance_driven"] = df.apply(lambda row: calculate_distance_driven(row), axis=1)
df = bin_distance(df)

df["distance_from_airport"] = df.apply(lambda row: calculate_distance_from_nearest_airport(row), axis=1)
df = bin_distance_from_airport(df)

df = one_hot_encode_pickup_time(df)

df.drop(["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"], inplace=True, axis=1)

save_clean_as_feather(df)

describe_data(df)

# violin_plot_feature_target(df, "fare_amount", "near_airport")

# correlation_matrix(df, "fare_amount")
