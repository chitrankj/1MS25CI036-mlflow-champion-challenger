import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"


def get_latest_run_by_name(client: MlflowClient, experiment_id: str, run_name: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No finished run named '{run_name}' found in '{EXPERIMENT_NAME}'. "
            "Run evaluate.py first!"
        )
    return runs[0]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run evaluate.py first!")

    run_a = get_latest_run_by_name(client, exp.experiment_id, "A_sentiment_distilbert-sst2")
    run_b = get_latest_run_by_name(client, exp.experiment_id, "B_sentiment_roberta-large")
    run_cmp = get_latest_run_by_name(client, exp.experiment_id, "compare_sentiment")

    winner_key = run_cmp.data.tags.get("winner_key", "B")
    print(f"Found dynamic runs -> A: {run_a.info.run_id}, B: {run_b.info.run_id}")
    print(f"Summary run winner_key: Model {winner_key}")

    # Ensure registered model exists
    try:
        client.create_registered_model(
            name=REGISTERED_MODEL_NAME,
            description="Sentiment Classification Champion-Challenger Registry",
        )
    except Exception:
        pass

    versions = {}
    for key, run_obj in [("A", run_a), ("B", run_b)]:
        model_uri = f"runs:/{run_obj.info.run_id}/sentiment_pyfunc"
        hf_model_id = run_obj.data.params.get("hf_model_id", "unknown")
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

        # Store hf_model_id and run_name as version tags so predict_from_mlflow.py can inspect them
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            key="hf_model_id",
            value=hf_model_id,
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            key="source_run_name",
            value=run_obj.data.tags.get("mlflow.runName", ""),
        )
        versions[key] = (mv.version, hf_model_id)
        print(f"Registered Model {key} ({hf_model_id}) as Version {mv.version}")

    # Assign @champion and @challenger aliases
    challenger_key = "B" if winner_key == "A" else "A"
    champ_ver, champ_hf = versions[winner_key]
    chall_ver, chall_hf = versions[challenger_key]

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME, alias="champion", version=champ_ver
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME, alias="challenger", version=chall_ver
    )

    print("\n=====================================================================")
    print(f"Alias '@champion'   -> Version {champ_ver} ({champ_hf})")
    print(f"Alias '@challenger' -> Version {chall_ver} ({chall_hf})")
    print("=====================================================================")


if __name__ == "__main__":
    main()