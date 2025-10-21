# Approximating LLM Decision-Making within a Markov Decision Process

This project investigates how the decision-making behavior of a **Large Language Model (LLM)** can be **approximated** within a **Markov Decision Process (MDP)** framework.  
The LLM receives structured states (e.g., traits and situational features) and outputs probabilistic action decisions.  
A smaller, trainable model (e.g., logistic regression, MLP, Transformer) is trained to imitate this mapping, enabling efficient simulation of LLM-like agents in computational environments.

---

## Project Overview

Formally, the LLM defines a conditional mapping:

$$
\pi_{\text{LLM}}: s \mapsto P(a \mid s)
$$
where $s$ denotes the **state** (a combination of traits and contextual features), and $a$ denotes the **action** ("buys ice cream").  

The objective is to train a lightweight **student model** $\pi_{\theta}$ to approximate:

$$
\pi_{\theta}(a \mid s) \approx \pi_{\text{LLM}}(a \mid s)
$$

This enables low-cost decision-making while preserving the behavioral patterns of the teacher LLM.

---

## Repository Structure

```
├── data/                         # Training and test datasets (JSONL)
├── models/
│   ├── saved_models/             # Trained model checkpoints (.pt)
│   └── configs/                  # Model configuration files
├── results/
│   ├── metrics/                  # Evaluation metrics (CSV files)
│   ├── plots/                    # Visualization outputs
│   └── logs/                     # Training logs and diagnostic output
├── notebooks/
│   ├── 01_gemma3_downloader-4b.ipynb
│   ├── 02_llm_stability_check.ipynb
│   ├── 03_generate_datasets.ipynb
│   ├── 04_train_models.ipynb
│   ├── 05_evaluate_models.ipynb
│   └── 06_visualization_results.ipynb
├── requirements.txt
└── README.md
```

---

## Notebooks Overview

### **01_gemma3_downloader-4b.ipynb**  
**Purpose:** Downloads the Gemma-3-4B model and tokenizer from Hugging Face into a local directory.  
This notebook ensures reproducible access to the same LLM version used in subsequent experiments.

---

### **02_llm_stability_check.ipynb — LLM Consistency and Stability Evaluation**

**Goal:** Quantify the **stochastic stability** and **output reproducibility** of a causal LLM under repeated inference.

- **Experimental Objective:**  
  Evaluate whether repeated runs of the same prompt produce stable numeric and semantic outputs:

  $$
  \pi_{\text{LLM}}^{(1)}(s) \approx \pi_{\text{LLM}}^{(2)}(s) \approx \ldots \approx \pi_{\text{LLM}}^{(N)}(s)
  $$
  where differences arise only from random seeds and inherent sampling noise.

- **Metrics:**
  - **Probability Stability:**
    - Per-prompt standard deviation
    - Mean Absolute Relative Difference (MARD)
    - Intraclass Correlation Coefficient (ICC)
  - **Semantic Stability:**
    - Mean cosine similarity of explanation embeddings

- **Outcome:**  
  Reports quantitative measures of how consistently the LLM produces probabilistic and textual outputs across identical inputs.

---

### **03_generate_datasets.ipynb — LLM-Based Behavioral Data Generation**

**Goal:** Generate a structured behavioral dataset representing LLM decisions over various trait–context combinations.

- **Formal Mapping:**

  $$
  (\text{traits}, \text{context}) \mapsto p_{\text{LLM}}(\text{buy ice cream})
  $$

- **Process:**
  1. Randomly sample valid combinations of traits and contexts.
  2. Insert them into a natural-language prompt template.
  3. Query the LLM deterministically (no sampling).
  4. Parse the JSON-formatted response:
     ```json
     {"buy": 0.74, "explanation": "Because it's a hot day and I like sweets."}
     ```
  5. Store examples as JSON Lines in:
     ```
     ../data/train.jsonl
     ../data/test.jsonl
     ```

- **Output:**  
  High-quality behavioral dataset suitable for supervised model training.

---

### **04_train_models.ipynb — Training Student Models**

**Goal:** Train smaller models to approximate the LLM’s decision probabilities using the generated dataset.

Implemented architectures:

| Model | Description | Order Sensitivity | Parameters |
|--------|--------------|-------------------|-------------|
| Logistic Regression | Single-layer linear regression with sigmoid | No | Few |
| Feedforward MLP | 2–4 hidden layers with ReLU activations | No | Medium |
| Encoder Transformer | Transformer encoder with self-attention | Yes (global) | Many |
| Decoder Transformer | GPT-like causal transformer with masking | Yes (causal) | Many |

- **Loss Function:**  
  $ \mathcal{L} = \frac{1}{N} \sum_n (\hat{y}_n - y_n)^2 $  
  or binary cross-entropy for probabilistic outputs.

- **Optimizer:** AdamW  
- **Outputs:**  
  - Training loss plots per epoch  
  - Model checkpoints stored in `../models/saved_models/`

---

### **05_evaluate_models.ipynb — Model Evaluation**

**Goal:** Evaluate trained models on the held-out test set.

- **Metrics:**
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Pearson/Spearman correlation between predicted and true probabilities
  - Optional calibration plots (predicted vs. target)

- **Outputs:**  
  Evaluation summaries stored as CSV in `../results/metrics/`.

---

### **06_visualization_results.ipynb — Visualization and Comparative Analysis**

**Goal:** Visualize and interpret model performance across architectures.

- **Visual Outputs:**
  - Training loss curves
  - Model comparison plots (MSE, correlation)
  - Error distributions and heatmaps

- **Outputs:**  
  Figures are saved in `../results/plots/`.

---

## Installation and Setup

### Requirements

The Python dependencies are listed in `requirements.txt`.  
Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure access to a GPU with CUDA if you plan to execute the Transformer models or the Gemma-3 LLM.

---

## Reproducible Workflow

1. **Download the base LLM:**  
   Run `01_gemma3_downloader-4b.ipynb`
2. **Check model stability:**  
   Run `02_llm_stability_check.ipynb`
3. **Generate behavioral data:**  
   Run `03_generate_datasets.ipynb`
4. **Train approximation models:**  
   Run `04_train_models.ipynb`
5. **Evaluate models:**  
   Run `05_evaluate_models.ipynb`
6. **Visualize results:**  
   Run `06_visualization_results.ipynb`

All notebooks are modular and can be executed independently.

---

## Research Significance

This project contributes to the emerging field of **LLM behavioral modeling**, demonstrating how probabilistic decision tendencies of large models can be replicated by compact architectures within an MDP framework.

Mathematically, the aim is to minimize:

$$
\mathbb{E}_{s \sim \mathcal{D}} \left[ \left( \pi_{\theta}(a \mid s) - \pi_{\text{LLM}}(a \mid s) \right)^2 \right]
$$
where $\mathcal{D}$ is the distribution of behavioral contexts.

Such approximations enable **interpretable**, **efficient**, and **simulation-compatible** agent models that reflect the statistical behavior of large foundation models.

---

## Output Artifacts

| Folder | Description |
|--------|--------------|
| `data/` | Raw and processed behavioral datasets (train/test JSONL) |
| `models/saved_models/` | Trained student model checkpoints |
| `models/configs/` | Hyperparameter and architecture configs |
| `results/metrics/` | Evaluation results (MSE, MAE, correlations) |
| `results/plots/` | Visualization outputs for comparison |
| `results/logs/` | Training and evaluation logs |

---
