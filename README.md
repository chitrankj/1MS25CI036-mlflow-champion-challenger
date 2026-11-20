# Champion–Challenger Model Comparison: Sentiment Classification

**Student USN:** `1MS25CI036`  
**Course:** Skill Lab (MLOps with MLflow) · Extension of Labs 4, 5, 8, 9, 10, 11 and 13

---

## 1. Task and Models

- **Task:** Binary Sentiment Classification (`sentiment-analysis`)
- **Dataset:** GLUE `nyu-mll/glue` (`sst2` validation split), fixed random subset of **200 samples** (`seed = 42`), saved to `data/eval_inputs.csv`.
- **Model A (Current Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters, fast binary baseline)
- **Model B (Challenger Candidate):** `siebert/sentiment-roberta-large-english` (~355M parameters, larger RoBERTa binary classifier)

---

## 2. How to Run

Run these commands in order from the project root directory in Windows PowerShell:

1. **Create and activate the virtual environment, then install dependencies:**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Start the MLflow Tracking Server (keep running in Terminal 1):**
   ```powershell
   python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
   ```

3. **Evaluate both models on the shared 200-sample subset and log to MLflow (in Terminal 2):**
   ```powershell
   python .\evaluate.py
   ```

4. **Register both models in the MLflow Model Registry and set `@champion` and `@challenger` aliases:**
   ```powershell
   python .\register_champion.py
   ```

5. **Load the Champion model via its Registry alias URI and predict on unseen data:**
   ```powershell
   python .\predict_from_mlflow.py
   ```

---

## 3. Champion–Challenger Decision Rule (Declared Before Running Experiments)

Model A (`distilbert-base-uncased-finetuned-sst-2-english`) is treated as the current **Champion**, and Model B (`siebert/sentiment-roberta-large-english`) is evaluated as the **Challenger**.

1. **Primary Metric:** `macro_f1` (Macro-averaged F1-score across `NEGATIVE` = 0 and `POSITIVE` = 1 classes).
2. **Minimum Improvement Margin:** The Challenger (`Model B`) must beat the Champion (`Model A`) on `macro_f1` by **at least +0.010 (+1.0 percentage point)** (`delta_primary_metric >= 0.010`). On a 200-sample evaluation set, a single sample flip alters accuracy/F1 by 0.5 percentage points (`0.005`), so requiring at least a 1.0 percentage point improvement (at least 2 net additional correct predictions) prevents promoting a 5x heavier model due to single-sample noise.
3. **Tie-Breaker and Latency Guardrail:**
   - **Latency Guardrail:** Reject any Challenger whose average inference latency (`avg_latency_ms`) is more than **5.0x** the Champion's latency (`latency_ratio_B_over_A > 5.0`), ensuring CPU deployment remains responsive.
   - **Tie-Breaker:** If `delta_primary_metric` falls within the margin (`-0.010 < delta_primary_metric < +0.010`), prefer the faster model (lower `avg_latency_ms`).
4. **Metric Direction:**
   - `macro_f1`, `accuracy`, `precision`, `recall`: **Higher is better**
   - `avg_latency_ms`, `total_inference_s`: **Lower is better**

---

## 4. Results Table

| Metric | Direction | Model A (`distilbert-sst2`) | Model B (`roberta-large`) | Difference (B − A) |
| :--- | :--- | :---: | :---: | :---: |
| **Macro-F1 (Primary)** | Higher is better | 0.9141 | 0.9344 | +0.0203 |
| **Accuracy** | Higher is better | 0.9150 | 0.9350 | +0.0200 |
| **Precision (Macro)** | Higher is better | 0.9183 | 0.9371 | +0.0188 |
| **Recall (Macro)** | Higher is better | 0.9121 | 0.9329 | +0.0208 |
| **Avg Latency (ms/sample)** | Lower is better | 34.52 | 392.73 | +358.21 (11.38x) |
| **Total Inference Time (s)** | Lower is better | 6.9036 | 78.5467 | +71.6431 |

---

## 5. Decision and Trade-off Discussion

Under our pre-declared Champion–Challenger decision rule, **Model A (`distilbert-base-uncased-finetuned-sst-2-english`) remains the Champion**, and **Model B (`siebert/sentiment-roberta-large-english`) is rejected by the latency guardrail** and assigned the `@challenger` alias.

On primary quality metrics, the Challenger (Model B) improved upon the Champion (Model A) across every classification measure: Macro-F1 increased by **+0.0203 (+2.03 percentage points)** from `0.9141` to `0.9344`, and overall accuracy rose by **+2.00 percentage points** from `0.9150` (183/200) to `0.9350` (187/200). While this `+0.0203` gain exceeds our required `+0.010` minimum improvement margin, on a sample size of `n = 200` a 2.0 percentage point difference represents only **4 net additional correct predictions** (primarily on movie reviews containing subtle contrast or mixed clauses). Given a binomial standard error of roughly $\pm 1.9\%$ at this sample size, a 4-sample improvement is modest.

More critically, the computational trade-off on CPU is severe. Because RoBERTa-large (~355M parameters) is over 5.3x larger than DistilBERT (~67M parameters), total inference time jumped from `6.9036 s` to `78.5467 s`, pushing average per-sample latency from `34.52 ms` to `392.73 ms`—an **11.38x slowdown**. Because `11.38x` breaches our pre-stated `5.0x` maximum latency guardrail, promoting Model B would degrade CPU service throughput by over an order of magnitude for only 4 extra correct predictions out of 200. Retaining Model A as `@champion` preserves fast CPU inference while maintaining 91.41% Macro-F1.

---

## 6. Limitations and Declaration

- **Limitations:** Evaluating on `n = 200` validation samples carries a sampling margin of error of roughly $\pm 1.9\%$, meaning small metric differences can fluctuate across random seeds. Furthermore, SST-2 consists of short movie review sentences, so findings may not generalize directly to long-form product or financial text, and latency ratios measured on CPU (`batch_size = 16`) would narrow significantly on a GPU with FP16 quantization.
- **Declaration:** I used an AI assistant to help structure the boilerplate MLflow PyFunc wrapper, plotting functions, and README formatting. I have reviewed, executed, and verified every line of code on my machine and am prepared to explain all design choices and results during the viva.
