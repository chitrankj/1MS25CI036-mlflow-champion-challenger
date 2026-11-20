import json
import os
import time
import warnings
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

warnings.filterwarnings("ignore")

# =====================================================================
# 5.1 CONFIGURATION & MLFLOW SETUP
# =====================================================================
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
STUDENT_USN = "1MS25CI036"  
SEED = 42
EVAL_SAMPLES = 200
MAX_LENGTH = 128
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if DEVICE == "cuda" else -1

# Pre-declared Champion-Challenger Rule parameters
PRIMARY_METRIC = "macro_f1"
MIN_MARGIN = 0.010          # +1.0 percentage point minimum gain
MAX_LATENCY_RATIO = 5.0     # Challenger latency cannot exceed 5x Champion

MODELS = {
    "A": {
        "run_name": "A_sentiment_distilbert-sst2",
        "short_name": "distilbert-sst2",
        "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion_candidate",
    },
    "B": {
        "run_name": "B_sentiment_roberta-large",
        "short_name": "roberta-large",
        "hf_model_id": "siebert/sentiment-roberta-large-english",
        "role": "challenger_candidate",
    },
}

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =====================================================================
# CUSTOM MLFLOW PYFUNC WRAPPER (Lab 9)
# =====================================================================
class HFSentimentPyFunc(mlflow.pyfunc.PythonModel):
    """
    Wraps a Hugging Face sentiment-analysis pipeline as an MLflow PyFunc model.
    When loaded via mlflow.pyfunc.load_model(), it accepts a DataFrame with a
    'text' column and returns a DataFrame with 'text', 'pred_label', 'pred_id',
    and 'confidence'.
    """

    def __init__(self, hf_model_id: str, max_length: int = 128):
        self.hf_model_id = hf_model_id
        self.max_length = max_length
        self.pipe = None

    def load_context(self, context):
        device_id = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.hf_model_id,
            device=device_id,
            truncation=True,
            max_length=self.max_length,
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(x) for x in model_input]

        if self.pipe is None:
            self.load_context(context)

        raw_preds = self.pipe(texts, truncation=True, max_length=self.max_length)
        pred_labels = []
        pred_ids = []
        confidences = []

        for res in raw_preds:
            lbl = res["label"].upper()
            is_pos = 1 if ("POS" in lbl or lbl == "LABEL_1") else 0
            pred_labels.append("POSITIVE" if is_pos == 1 else "NEGATIVE")
            pred_ids.append(is_pos)
            confidences.append(round(float(res["score"]), 4))

        return pd.DataFrame(
            {
                "text": texts,
                "pred_label": pred_labels,
                "pred_id": pred_ids,
                "confidence": confidences,
            }
        )


# =====================================================================
# 5.2 LOAD DATASET & SAVE SHARED INPUTS
# =====================================================================
def load_and_save_eval_data() -> pd.DataFrame:
    eval_path = "data/eval_inputs.csv"
    if os.path.exists(eval_path):
        print(f"Loading existing shared evaluation dataset from {eval_path}...")
        return pd.read_csv(eval_path)

    print("Downloading GLUE SST-2 validation split and sampling 200 items...")
    ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    ds_sampled = ds.shuffle(seed=SEED).select(range(EVAL_SAMPLES))

    df = pd.DataFrame(
        {
            "id": range(EVAL_SAMPLES),
            "text": ds_sampled["sentence"],
            "reference_id": ds_sampled["label"],
            "reference_label": [
                "POSITIVE" if lbl == 1 else "NEGATIVE"
                for lbl in ds_sampled["label"]
            ],
        }
    )
    df.to_csv(eval_path, index=False)
    print(f"Saved shared evaluation inputs ({len(df)} rows) to {eval_path}")
    return df


# =====================================================================
# PLOT GENERATORS (5.5)
# =====================================================================
def save_confusion_matrix_plot(y_true, y_pred, short_name: str) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["NEGATIVE (0)", "POSITIVE (1)"]
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix: {short_name}")
    plt.tight_layout()
    out_path = f"outputs/confusion_matrix_{short_name}.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def save_comparison_bar_chart(metrics_a: dict, metrics_b: dict) -> str:
    metric_keys = ["accuracy", "macro_f1", "precision", "recall"]
    a_vals = [metrics_a[k] for k in metric_keys]
    b_vals = [metrics_b[k] for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    rects1 = ax.bar(x - width / 2, a_vals, width, label="Model A (DistilBERT)")
    rects2 = ax.bar(x + width / 2, b_vals, width, label="Model B (RoBERTa-Large)")

    ax.set_ylabel("Score (Higher is Better)")
    ax.set_title("Champion (Model A) vs. Challenger (Model B) Evaluation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1", "Precision", "Recall"])
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc="lower right")

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = "outputs/metrics_comparison_bar.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


# =====================================================================
# MAIN EVALUATION & LOGGING PIPELINE
# =====================================================================
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    eval_df = load_and_save_eval_data()
    texts = eval_df["text"].tolist()
    y_true = eval_df["reference_id"].tolist()

    lib_versions = (
        f"transformers={transformers.__version__}, "
        f"torch={torch.__version__}, "
        f"mlflow={mlflow.__version__}, "
        f"sklearn={sklearn.__version__}"
    )

    all_metrics = {}
    all_preds_df = {}
    run_ids = {}

    for key, cfg in MODELS.items():
        print(f"\n--- Running {cfg['run_name']} ({cfg['hf_model_id']}) ---")
        clf = pipeline(
            "sentiment-analysis",
            model=cfg["hf_model_id"],
            device=DEVICE_ID,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Warm-up pass on 2 examples so model loading isn't counted in latency
        _ = clf(texts[:2])

        start_t = time.perf_counter()
        raw_outputs = clf(
            texts, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH
        )
        total_time_s = time.perf_counter() - start_t
        avg_latency_ms = (total_time_s / len(texts)) * 1000.0

        y_pred = []
        pred_labels = []
        confidences = []
        for out in raw_outputs:
            lbl = out["label"].upper()
            p_id = 1 if ("POS" in lbl or lbl == "LABEL_1") else 0
            y_pred.append(p_id)
            pred_labels.append("POSITIVE" if p_id == 1 else "NEGATIVE")
            confidences.append(round(float(out["score"]), 4))

        # Compute task metrics
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        prec = precision_score(y_true, y_pred, average="macro")
        rec = recall_score(y_true, y_pred, average="macro")

        metrics_dict = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "avg_latency_ms": round(float(avg_latency_ms), 2),
            "total_inference_s": round(float(total_time_s), 4),
        }
        all_metrics[key] = metrics_dict

        # Save per-model predictions CSV
        pred_df = eval_df.copy()
        pred_df["prediction_id"] = y_pred
        pred_df["prediction_label"] = pred_labels
        pred_df["confidence"] = confidences
        pred_df["correct"] = pred_df["reference_id"] == pred_df["prediction_id"]

        pred_csv_path = f"outputs/predictions_{cfg['short_name']}.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        all_preds_df[key] = pred_df

        # Save confusion matrix plot
        cm_path = save_confusion_matrix_plot(y_true, y_pred, cfg["short_name"])

        # Log run in MLflow (Section 5.3)
        with mlflow.start_run(run_name=cfg["run_name"]) as run:
            run_ids[key] = run.info.run_id

            # Log parameters
            mlflow.log_params(
                {
                    "hf_model_id": cfg["hf_model_id"],
                    "task": "sentiment-analysis",
                    "role": cfg["role"],
                    "max_length": MAX_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "eval_samples": EVAL_SAMPLES,
                    "seed": SEED,
                    "device": DEVICE,
                }
            )

            # Log metrics
            mlflow.log_metrics(metrics_dict)

            # Log tags
            mlflow.set_tags(
                {
                    "student_usn": STUDENT_USN,
                    "library_versions": lib_versions,
                    "dataset_name": "glue/sst2 (validation)",
                    "metric_direction_f1": "higher_is_better",
                    "metric_direction_latency": "lower_is_better",
                }
            )

            # Log custom PyFunc model
            input_example = pd.DataFrame({"text": texts[:2]})
            pyfunc_model = HFSentimentPyFunc(
                hf_model_id=cfg["hf_model_id"], max_length=MAX_LENGTH
            )
            mlflow.pyfunc.log_model(
                artifact_path="sentiment_pyfunc",
                python_model=pyfunc_model,
                input_example=input_example,
            )

            # Log artifacts for this model run
            mlflow.log_artifact(pred_csv_path)
            mlflow.log_artifact("data/eval_inputs.csv")
            mlflow.log_artifact(cm_path)

            print(
                f"Logged {cfg['run_name']} (Run ID: {run.info.run_id}) | "
                f"Macro-F1: {metrics_dict['macro_f1']:.4f} | "
                f"Avg Latency: {metrics_dict['avg_latency_ms']:.2f} ms"
            )

    # =====================================================================
    # 5.4 & 5.5 CHAMPION-CHALLENGER DECISION & SUMMARY RUN
    # =====================================================================
    m_a = all_metrics["A"]
    m_b = all_metrics["B"]
    delta_f1 = round(m_b[PRIMARY_METRIC] - m_a[PRIMARY_METRIC], 4)
    latency_ratio = round(m_b["avg_latency_ms"] / max(m_a["avg_latency_ms"], 1e-6), 2)

    # Apply pre-declared Champion-Challenger Rule
    if latency_ratio > MAX_LATENCY_RATIO:
        winner_key = "A"
        reason = (
            f"Challenger rejected by latency guardrail "
            f"({latency_ratio}x > {MAX_LATENCY_RATIO}x)."
        )
    elif delta_f1 >= MIN_MARGIN:
        winner_key = "B"
        reason = (
            f"Challenger beat Champion on {PRIMARY_METRIC} by {delta_f1:+.4f} "
            f"(>= {MIN_MARGIN}) within latency guardrail ({latency_ratio}x)."
        )
    elif delta_f1 <= -MIN_MARGIN:
        winner_key = "A"
        reason = f"Champion outperformed Challenger by {-delta_f1:+.4f}."
    else:
        # Tie-breaker: within margin -> faster model wins
        winner_key = (
            "A" if m_a["avg_latency_ms"] <= m_b["avg_latency_ms"] else "B"
        )
        reason = (
            f"Difference ({delta_f1:+.4f}) within margin (+/-{MIN_MARGIN}); "
            f"faster model ({MODELS[winner_key]['short_name']}) selected."
        )

    loser_key = "B" if winner_key == "A" else "A"

    # Save metrics_summary.json
    summary_payload = {
        "experiment": EXPERIMENT_NAME,
        "eval_samples": EVAL_SAMPLES,
        "primary_metric": PRIMARY_METRIC,
        "min_margin": MIN_MARGIN,
        "max_latency_ratio": MAX_LATENCY_RATIO,
        "delta_primary_metric": delta_f1,
        "latency_ratio_B_over_A": latency_ratio,
        "winner_key": winner_key,
        "winner_run_name": MODELS[winner_key]["run_name"],
        "winner_hf_model_id": MODELS[winner_key]["hf_model_id"],
        "decision_reason": reason,
        "models": {
            "A": {**MODELS["A"], "run_id": run_ids["A"], "metrics": m_a},
            "B": {**MODELS["B"], "run_id": run_ids["B"], "metrics": m_b},
        },
    }
    summary_json_path = "outputs/metrics_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Save comparison bar chart
    bar_chart_path = save_comparison_bar_chart(m_a, m_b)

    # Save 5 qualitative disagreement cases (Section 5.5)
    df_a = all_preds_df["A"]
    df_b = all_preds_df["B"]
    disagree_mask = df_a["prediction_id"] != df_b["prediction_id"]
    disagree_df = pd.DataFrame(
        {
            "id": df_a.loc[disagree_mask, "id"],
            "text": df_a.loc[disagree_mask, "text"],
            "reference_label": df_a.loc[disagree_mask, "reference_label"],
            "model_A_pred": df_a.loc[disagree_mask, "prediction_label"],
            "model_A_conf": df_a.loc[disagree_mask, "confidence"],
            "model_B_pred": df_b.loc[disagree_mask, "prediction_label"],
            "model_B_conf": df_b.loc[disagree_mask, "confidence"],
        }
    ).head(5)
    disagree_df["comment"] = [
        "Nuanced or contrasting sentiment clause where the two architectures diverge."
        for _ in range(len(disagree_df))
    ]
    disagree_csv_path = "outputs/disagreements_top5.csv"
    disagree_df.to_csv(disagree_csv_path, index=False)

    # Log summary comparison run: compare_sentiment
    with mlflow.start_run(run_name="compare_sentiment"):
        mlflow.log_params(
            {
                "primary_metric": PRIMARY_METRIC,
                "margin": MIN_MARGIN,
                "max_latency_ratio": MAX_LATENCY_RATIO,
                "model_A_id": MODELS["A"]["hf_model_id"],
                "model_B_id": MODELS["B"]["hf_model_id"],
                "eval_samples": EVAL_SAMPLES,
            }
        )
        mlflow.log_metrics(
            {
                "delta_primary_metric": delta_f1,
                "A_macro_f1": m_a["macro_f1"],
                "B_macro_f1": m_b["macro_f1"],
                "A_accuracy": m_a["accuracy"],
                "B_accuracy": m_b["accuracy"],
                "A_avg_latency_ms": m_a["avg_latency_ms"],
                "B_avg_latency_ms": m_b["avg_latency_ms"],
                "latency_ratio_B_over_A": latency_ratio,
            }
        )
        mlflow.set_tags(
            {
                "student_usn": STUDENT_USN,
                "winner": MODELS[winner_key]["run_name"],
                "winner_key": winner_key,
                "winner_hf_model_id": MODELS[winner_key]["hf_model_id"],
                "challenger_key": loser_key,
                "decision_reason": reason,
            }
        )

        # Log all summary artifacts (Section 5.5)
        mlflow.log_artifact(summary_json_path)
        mlflow.log_artifact(bar_chart_path)
        mlflow.log_artifact(disagree_csv_path)
        mlflow.log_artifact("data/eval_inputs.csv")
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")

    print("\n=====================================================================")
    print(f"WINNER: Model {winner_key} ({MODELS[winner_key]['hf_model_id']})")
    print(f"REASON: {reason}")
    print(f"Summary saved to {summary_json_path}")
    print("=====================================================================")


if __name__ == "__main__":
    main()