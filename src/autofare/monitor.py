import hashlib
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from autofare.clean import clean
from autofare.train import FEATURES, NUM, TARGET, build_model

V2_CONFIG = "gbr"
PROMOTE_IF_BETTER_BY = 0.08


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def mae_by_group(model, df, group="zone"):
    out = {}
    for g, part in df.groupby(group):
        out[g] = float(mean_absolute_error(part[TARGET], model.predict(part[FEATURES])))
    return out


def main():
    params = yaml.safe_load(open("params.yaml"))
    seed = params["model"]["random_state"]
    reg_name = params["mlflow"]["registered_model_name"]

    train_path = params["data"]["train_path"]
    train = clean(pd.read_csv(train_path))
    later = pd.read_csv(params["data"]["later_path"])
    holdout = pd.read_csv(params["data"]["holdout_path"])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    if len(client.search_model_versions(f"name='{reg_name}'")) >= 2:
        raise SystemExit("Version 2 already exists. Do not run this twice.")

    psi_values = {c: round(psi(train[c], later[c]), 4) for c in NUM}
    most_drifted = max(psi_values, key=psi_values.get)
    print("PSI:", psi_values)
    print("Most drifted column:", most_drifted)

    v1 = joblib.load("models/model_v1.joblib")
    zone_mae = {z: round(m, 3) for z, m in mae_by_group(v1, later).items()}
    worst = max(zone_mae, key=zone_mae.get)
    print("v1 MAE per zone:", zone_mae)
    print("Worst served zone:", worst)

    later = later.sort_values(["date", "record_id"]).reset_index(drop=True)
    cut = int(len(later) * 0.7)
    first, last = later.iloc[:cut], later.iloc[cut:]

    train_v2 = pd.concat([train[FEATURES + [TARGET]], first[FEATURES + [TARGET]]], ignore_index=True)
    cfg = next(c for c in params["experiments"] if c["name"] == V2_CONFIG)
    v2 = build_model(cfg, seed)
    v2.fit(train_v2[FEATURES], train_v2[TARGET])

    mae_v1 = float(mean_absolute_error(last[TARGET], v1.predict(last[FEATURES])))
    mae_v2 = float(mean_absolute_error(last[TARGET], v2.predict(last[FEATURES])))
    improvement = (mae_v1 - mae_v2) / mae_v1
    promote = improvement >= PROMOTE_IF_BETTER_BY
    decision = "promote" if promote else "keep"
    print(f"v1 MAE on last 30%: {mae_v1:.3f}")
    print(f"v2 MAE on last 30%: {mae_v2:.3f}")
    print(f"Improvement: {improvement * 100:.1f}%  ->  decision: {decision}")

    exp_name = "autofare-retrain"
    if client.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=Path("mlartifacts").resolve().as_uri())
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_param("data_sha256", hashlib.sha256(open(train_path, "rb").read()).hexdigest())
        mlflow.log_params({k: str(v) for k, v in cfg.items()})
        mlflow.log_metric("mae_v1_check", mae_v1)
        mlflow.log_metric("mae_v2_check", mae_v2)
        mlflow.log_metric("mae_improvement", improvement)
        mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", reg_name)
    print("Registered version", mv.version)

    if promote:
        client.set_registered_model_alias(reg_name, "champion", mv.version)
        joblib.dump(v2, "models/model.joblib")
        champion = v2
    else:
        champion = v1
    print("Champion is now version", mv.version if promote else 1)

    pred = champion.predict(holdout[FEATURES])
    Path("predictions").mkdir(exist_ok=True)
    out = pd.DataFrame({"record_id": holdout["record_id"], "prediction": np.round(pred, 2)})
    out.to_csv("predictions/holdout_predictions.csv", index=False)
    print("Saved predictions/holdout_predictions.csv with", len(pred), "rows")

    path = Path("reports/findings.yaml")
    findings = yaml.safe_load(open(path))
    findings["monitoring"] = {
        "psi": psi_values,
        "most_drifted_feature": most_drifted,
        "mae_by_group": zone_mae,
        "worst_served_group": worst,
        "mae_v1_on_check": round(mae_v1, 3),
        "mae_v2_on_check": round(mae_v2, 3),
        "decision": decision,
    }
    if promote:
        findings["tracking"]["champion_version"] = int(mv.version)
    with open(path, "w") as f:
        yaml.safe_dump(findings, f, sort_keys=False)
    print("Updated reports/findings.yaml")


if __name__ == "__main__":
    main()
