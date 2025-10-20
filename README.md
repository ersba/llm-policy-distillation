# 🧠 LLM Decision Approximation

## Overview

This project investigates how the decision-making behavior of a **Large Language Model (LLM)** can be **approximated** within a **Markov Decision Process (MDP)**.  
The LLM receives states (e.g., traits and situational features) as input and produces action probabilities.  
A smaller, trainable model (e.g., MLP, Transformer) is used to approximate this behavior to enable efficient decision-making in simulation environments, such as agent-based systems.

---

## 📁 Repository Structure

```
llm-decision-approximation/
│
├── data/
│   ├── raw/                 # Raw LLM outputs or prompt results
│   ├── processed/           # Preprocessed train/test datasets
│   ├── prompts/             # Prompt templates for data generation
│   └── README.md
│
├── notebooks/
│   ├── 00_setup_environment.ipynb     # Environment setup & package checks
│   ├── 01_generate_datasets.ipynb     # Generate training/test data with LLM
│   ├── 02_train_models.ipynb          # Train approximation models
│   ├── 03_evaluate_models.ipynb       # Evaluate and compare architectures
│   ├── 04_ablation_studies.ipynb      # Perform ablation and sensitivity studies
│   └── 05_visualization_results.ipynb # Visualize results and metrics
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py        # Data loading, preprocessing, splitting
│   ├── model_utils.py       # Model definitions, training loops, losses
│   ├── eval_utils.py        # Metrics, plotting, and evaluation
│   └── config.py            # Global paths and hyperparameters
│
├── models/
│   ├── saved_models/        # Trained model checkpoints
│   └── configs/             # Model configuration files
│
├── results/
│   ├── metrics/             # Evaluation metrics (CSV)
│   ├── plots/               # Plots and visualizations
│   └── logs/                # Training logs and outputs
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/llm-decision-approximation.git
cd llm-decision-approximation
```

### 2️⃣ Set up the environment
Using **conda**:
```bash
conda env create -f environment.yml
conda activate llm-decision-approximation
```

or using **pip**:
```bash
python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate         # (Windows)
pip install -r requirements.txt
```

---

## 🧪 Notebooks Workflow

| Step | Notebook | Description |
|------|-----------|-------------|
| **0. Setup** | `00_setup_environment.ipynb` | Verify dependencies, GPU/CUDA availability, and setup |
| **1. Data Generation** | `01_generate_datasets.ipynb` | Generate train/test data via LLM (e.g., DeepSeek-R1, Llama) |
| **2. Training** | `02_train_models.ipynb` | Train multiple architectures (LogReg, MLP, Transformer) |
| **3. Evaluation** | `03_evaluate_models.ipynb` | Compute metrics such as KL, CE, ECE, and FQE |
| **4. Ablation** | `04_ablation_studies.ipynb` | Analyze influence of traits, inputs, and model design |
| **5. Visualization** | `05_visualization_results.ipynb` | Produce figures and plots for reports or publications |

---

## 🧱 Source Module Overview

| File | Purpose |
|------|----------|
| `data_utils.py` | Load, preprocess, and split datasets |
| `model_utils.py` | Define architectures and training loops |
| `eval_utils.py` | Compute metrics, visualize results |
| `config.py` | Centralized configuration and paths |
| `__init__.py` | Makes `src/` importable as a Python package |

---

## 📊 Key Evaluation Metrics

- **Cross-Entropy / KL-Divergence** – Measures similarity between LLM and approximated policy  
- **Expected Calibration Error (ECE)** – Evaluates probability calibration  
- **Fitted Q Evaluation (FQE)** – Offline estimate of policy quality  
- **Ablation metrics** – Effect of traits, context, or architecture variants

---

## 🧠 Research Objective

The repository supports a multi-phase research pipeline:

1. **Basic Project:** Develop and evaluate the foundational approximation architecture  
2. **Main Project:** Integrate the model into a Multi-Agent Simulation (e.g., Christmas market scenario)  
3. **Master’s Thesis:** Extend to a complex urban simulation validated with population data

---

## 🧩 Technologies Used

- **Python ≥ 3.10**
- **PyTorch**
- **Hugging Face Transformers**
- **Pandas, NumPy, scikit-learn**
- **Matplotlib, Seaborn**
- **Jupyter / JupyterLab**

---

## 📈 Usage Example

Launch Jupyter Lab:
```bash
jupyter lab notebooks/
```

Open `02_train_models.ipynb` and run:
```python
from src.model_utils import MLPPolicy, train_model
from src.data_utils import load_dataset
```

---

## 📄 License

MIT License © 2025 [Your Name]  
This project may be used and extended for research purposes.
