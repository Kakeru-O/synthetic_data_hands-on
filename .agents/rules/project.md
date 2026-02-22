---
trigger: always_on
---

# Synthetic Data Hands-on

## 1. Project Overview
This project is a hands-on demonstration designed for engineers to experience the power of **Synthetic Data** and **Modern Python Tooling**.

### The Scenario
The Marketing Department wants to share customer data with an external AI analysis vendor to improve campaign targeting. However, the Legal Department has blocked the request due to privacy concerns (GDPR/APPI).
**Your Goal:** Create a synthetic dataset that preserves the statistical properties of the original customer data but contains **zero real individuals**, proving to Legal that it is safe to share.

### Tech Stack
* **Language:** Python 3.10+
* **Notebook/UI:** [marimo](https://github.com/marimo-team/marimo) (Reactive Python Notebook)
* **Package Manager:** [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer)
* **Synthetic Engine:** [MostlyAI SDK](https://mostly.ai/) (Free tier usage)
* **Data Handling:** pandas, altair (for visualization), scikit-learn (for nearest neighbor search)
* **Dataset:** Bank Marketing Data Set (UCI Machine Learning Repository)

---

## 2. Hands-on Flow (Application Logic)

The `app.py` is a fully interactive marimo notebook. Users will execute the cells from top to bottom to complete the simulation.


### Step 1: Data Ingestion ("The Dangerous Data")
* **Data Source:** Automatically download/load the "Bank Marketing" dataset (e.g., specific columns: `age`, `job`, `balance`, `housing`, `loan`, `duration`, `campaign`, `y`).
* **UI:** Display the raw dataframe using `marimo.ui.table`.
* **Context:** Show a markdown explanation highlighting that this data contains sensitive personal attributes and cannot be exported as-is.

### Step 2: Synthetic Data Generation
* **UI Controls:**
    * `sample_size_slider`: A slider to select the number of synthetic samples to generate (Range: 100 - 5,000, Default: 1,000).
    * `generate_button`: A button to trigger the generation process.
* **Logic:**
    * Upon button click, use the MostlyAI SDK to train a generator on the original data and seed a new synthetic dataset.
    * Show a progress indicator (spinner) during the API call.
    * Return a dataframe of synthetic data.

### Step 3: Statistical Validation ("The Proof")
* **Goal:** Visually prove that the synthetic data mimics the real data's distribution.
* **UI Controls:**
    * `column_selector`: A dropdown menu to select a column to visualize (e.g., 'age', 'balance', 'job').
* **Visualization:**
    * Use **Altair** to plot histograms/bar charts.
    * **Overlay** the "Original" (Blue) and "Synthetic" (Red) distributions on the same chart to show the alignment.
    * The chart must update reactively when the `column_selector` changes.

### Step 4: Privacy Verification
* **Goal:** Verify that the synthetic records are not copies of real people.
* **Logic:** Pick a random sample from the synthetic dataset and find its "Nearest Neighbor" in the original dataset (using Euclidean distance for numerical columns or simple matching for categorical).
* **UI:** Display a side-by-side comparison of the "Target Synthetic User" vs. the "Closest Real User".
* **Message:** Highlight the differences (e.g., "Closest match differs by X years in age and Y in balance") to demonstrate that no specific individual has been leaked (Re-identification risk assessment).

---

## 3. Installation & Usage

This project uses `uv` for dependency management.

### Setup
```bash
# 1. Clone the repository
git clone <repository-url>
cd <repository-name>

# 2. Sync dependencies (creates .venv automatically)
uv sync
```

### Run the App
Start the marimo server to open the interactive notebook in your browser.
```
Bash
uv run marimo edit app.py
```