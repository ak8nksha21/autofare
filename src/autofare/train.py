import hashlib

import joblib
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from autofare.clean import clean

NUM = ["distance_km", "wait_min", "passengers", "hour", "luggage_bags", "fuel_price_rs"]
CAT = ["booking", "zone"]
FEATURES = NUM + CAT
TARGET = "fare_rs"


def build_model(cfg, seed):
    if cfg["type"] == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            random_state=seed, n_jobs=-1)
    else:
        reg = GradientBoostingRegressor(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            random_state=seed)
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        remainder="passthrough")
    return Pipeline([("pre", pre), ("reg", reg)])


def main():
    params = yaml.safe_load(open("params.yaml"))
    seed = params["model"]["random_state"]
    train_path = params["data"]["train_path"]

    data_sha256 = hashlib.sha256(open(train_path, "rb").read()).hexdigest()

    train = clean(pd.read_csv(train_path))
    later = pd.read_csv(params["data"]["later_path"])  # already clean: do NOT call clean()

    X_train, y_train = train[FEATURES], train[TARGET]
    X_later, y_later = later[FEATURES], later[TARGET]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    best = {"mae": float("inf"), "run_id": None, "model": None}
    for cfg in params["experiments"]:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            model = build_model(cfg, seed)
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_later, model.predict(X_later))

            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_params({k: str(v) for k, v in cfg.items()})
            mlflow.log_metric("mae_later", mae)
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

            print(f"{cfg['name']}: mae_later={mae:.3f}  run_id={run.info.run_id}")
            if mae < best["mae"]:
                best = {"mae": mae, "run_id": run.info.run_id, "model": model}

    # register only the best run: this becomes version 1 (v1) and gets the champion alias
    name = params["mlflow"]["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)
    joblib.dump(best["model"], "models/model.joblib")

    print("\nBEST run_id:", best["run_id"])
    print("BEST mae_later:", round(best["mae"], 3))
    print("registered:", name, "version", mv.version, "alias champion")


if __name__ == "__main__":
    main()
