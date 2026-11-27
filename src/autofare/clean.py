import numpy as np
import pandas as pd


def clean(df):
    df = df.copy()

    # 1. Unit mixup: Station Area distance_km is in metres, so divide by 1000.
    mask = df["zone"] == "Station Area"
    df.loc[mask, "distance_km"] = df.loc[mask, "distance_km"] / 1000

    # 2. Sentinel: 9999 in wait_min is a fake value. Make it blank, then fill with the median.
    df["wait_min"] = df["wait_min"].replace(9999, np.nan)
    df["wait_min"] = df["wait_min"].fillna(df["wait_min"].median())

    # 3. Spelling: tidy booking into 3 clean categories.
    df["booking"] = df["booking"].str.strip().str.lower().str.rstrip(".")
    df["booking"] = df["booking"].replace({"sta": "stand", "str": "street"})

    # 4. Leak: drop the column that is only known after the ride.
    df = df.drop(columns=["fare_paid_receipt_rs"])

    return df


if __name__ == "__main__":
    train = pd.read_csv("data/raw/train.csv")
    out = clean(train)
    print(out.describe())
    print(out["booking"].value_counts())
