import pandas as pd

df = pd.read_csv("data/raw/train.csv")
hold = pd.read_csv("data/raw/holdout_features.csv")

print("1. Leaking column")
print("Not in holdout:", set(df.columns) - set(hold.columns))
print("Correlation:", df["fare_paid_receipt_rs"].corr(df["fare_rs"]))

print("2. Unit mixup")
print(df.groupby("zone")["distance_km"].agg(["min", "mean", "max"]))

print("3. Sentinel")
print(df["wait_min"].value_counts().head())

print("4. Duplicates")
print(df.duplicated().sum())

print("6. Target typos")
print(df["fare_rs"].sort_values().tail(10))

print("7. Spelling")
print(df["booking"].value_counts())

print("3b. Sentinel count")
print((df["wait_min"] == 9999).sum())

print("5. Stuck sensor: longest run of identical values")
d = df.copy()
d["date"] = pd.to_datetime(d["date"])
d = d.sort_values(["date", "record_id"])
for c in ["distance_km", "wait_min", "passengers", "hour", "luggage_bags", "fuel_price_rs"]:
    run = (d[c] != d[c].shift()).cumsum()
    print(c, run.value_counts().max())
