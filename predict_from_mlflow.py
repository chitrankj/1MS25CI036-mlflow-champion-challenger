import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-model"

# 8 brand-new examples not present in GLUE SST-2
NEW_EXAMPLES = [
    "The new lab server setup at MSRIT was surprisingly fast and effortless to configure.",
    "I waited two hours for tech support, only for the application to crash again immediately.",
    "Although the opening lecture felt slow, the hands-on MLOps demo made everything click.",
    "The battery drains in three hours and the keyboard feels flimsy for the price.",
    "This cafeteria masala dosa is hands down the best breakfast on campus.",
    "The documentation looks thorough at first glance, yet every code example throws a deprecation error.",
    "Deploying models with registry aliases completely eliminated our manual config bugs.",
    "Not a single feature promised in the release notes actually works as advertised.",
]


def ensure_new_data_csv(path: str = "data/new_data.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df = pd.DataFrame({"id": range(1, len(NEW_EXAMPLES) + 1), "text": NEW_EXAMPLES})
        df.to_csv(path, index=False)
        print(f"Created new unseen dataset at {path} ({len(df)} examples)")
    return pd.read_csv(path)


def load_model_by_alias(client: MlflowClient, alias: str):
    mv = client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME, alias=alias)
    hf_model_id = mv.tags.get("hf_model_id")
    if not hf_model_id:
        run = client.get_run(mv.run_id)
        hf_model_id = run.data.params.get("hf_model_id", "unknown")

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return loaded_model, mv.version, hf_model_id, model_uri


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    new_df = ensure_new_data_csv("data/new_data.csv")

    # Load Champion via alias URI
    champ_model, champ_ver, champ_hf_id, champ_uri = load_model_by_alias(client, "champion")
    print("\n=====================================================================")
    print(f"Loaded Champion URI : {champ_uri}")
    print(f"Registry Version    : v{champ_ver}")
    print(f"Hugging Face ID     : {champ_hf_id}")
    print("=====================================================================")

    champ_preds = champ_model.predict(new_df[["text"]])

    out_df = pd.DataFrame(
        {
            "id": new_df["id"],
            "text": new_df["text"],
            "champion_version": f"v{champ_ver}",
            "champion_hf_model": champ_hf_id,
            "champion_pred": champ_preds["pred_label"],
            "champion_confidence": champ_preds["confidence"],
        }
    )

    # Optionally load Challenger for side-by-side verification
    try:
        chall_model, chall_ver, chall_hf_id, _ = load_model_by_alias(client, "challenger")
        chall_preds = chall_model.predict(new_df[["text"]])
        out_df["challenger_version"] = f"v{chall_ver}"
        out_df["challenger_pred"] = chall_preds["pred_label"]
        out_df["challenger_confidence"] = chall_preds["confidence"]
    except Exception:
        pass

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/new_data_predictions.csv"
    out_df.to_csv(out_path, index=False)

    # Print clean formatted table for screenshot #3
    display_df = out_df[["id", "champion_pred", "champion_confidence", "text"]].copy()
    display_df["text"] = display_df["text"].str.slice(0, 62) + "..."
    print("\nPredictions on Unseen Data (via MLflow @champion alias):")
    print(display_df.to_string(index=False))
    print(f"\nSaved full predictions to {out_path}")


if __name__ == "__main__":
    main()