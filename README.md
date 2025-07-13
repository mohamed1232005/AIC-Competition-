# 🧠 EEG Signal Classification for Brain-Computer Interfaces (BCI)

This repository presents a solution for classifying Electroencephalography (EEG) signals tailored for Brain-Computer Interface (BCI) applications. Our focus lies on distinguishing between the two primary EEG paradigms:

- **Steady-State Visual Evoked Potentials (SSVEP)**
- **Motor Imagery (MI)**

By processing multi-channel EEG recordings, extracting key features, and leveraging machine learning algorithms, we aim to contribute to robust, non-invasive BCI systems for real-world communication and control use cases.

---

## 📁 Dataset Overview

The project uses the **mtcaic3 dataset**, comprising multi-channel EEG data. The structure includes:

- `train.csv`, `validation.csv`, and `test.csv`: metadata files referencing individual EEG trials.
- Each referenced file (e.g., `EEGdata.csv`) contains **raw EEG signals** for a single trial.
- Data is categorized into **MI** and **SSVEP** trials.

---

## 🔍 Methodology

We employ a **multi-stage pipeline** consisting of preprocessing, feature extraction, model training, and evaluation.

---

### 1. 🧹 Data Loading & Preprocessing

Each EEG trial is dynamically loaded using metadata.

- **Bandpass Filtering**: A 1–40 Hz Butterworth filter removes artifacts and preserves relevant EEG frequencies [1].
- **Baseline Correction**: Mitigates DC drifts and slow fluctuations.
- **Channel Selection**: Selects relevant channels using domain-specific knowledge for BCI.
- **Epoching**: Assumed to be pre-applied or handled implicitly during feature extraction.

---

### 2. 📊 Feature Extraction

We extract **frequency**, **time-domain**, **connectivity**, and **spatial** features:

#### A. Power Spectral Density (PSD) – via Welch’s Method

- Frequency bands: Delta (1–4 Hz), Theta (4–8 Hz), Alpha (8–12 Hz), Beta (12–30 Hz), Gamma (30–40 Hz).
- Captures oscillatory power over brainwave frequency bands.

#### B. Time-Domain Features

Statistical metrics computed per EEG channel:
- Mean, Standard Deviation
- Maximum, Minimum
- Kurtosis, Skewness

#### C. Functional Connectivity

- **Pearson Correlation Matrix** across EEG channels.
- Unique upper-triangle values represent inter-channel connectivity.

#### D. Common Spatial Pattern (CSP)

- Applied **only to MI data**.
- Enhances class separability by maximizing variance between classes [2].

---

### 3. 🤖 Model Training

We use **task-specific classifiers** for optimal performance:

#### A. Motor Imagery (MI)

- **Model**: `RandomForestClassifier`
- **Preprocessing**: Features normalized using `StandardScaler`

#### B. Steady-State Visual Evoked Potentials (SSVEP)

- **Model**: `XGBClassifier` (Extreme Gradient Boosting)
- **Preprocessing**: `StandardScaler` for normalization

Both classifiers use `LabelEncoder` to convert class labels into a numerical format.

---

### 4. 🧪 Evaluation & Submission

- Evaluation Metric: **Mean classification accuracy** (averaged across SSVEP and MI).
- The trained models are evaluated on a **held-out validation set**.
- Final predictions are saved to `submission.csv`, adhering to competition formatting.

---

## 🛠️ Dependencies

Ensure the following Python libraries are installed:


pip install numpy pandas scipy matplotlib seaborn scikit-learn mne xgboost


## 🚀 How to Run
1. Set up the Dataset
Ensure the mtcaic3 dataset is downloaded and located in:

BASE_PATH = "/kaggle/input/mtcaic3/"  # or local equivalent
It should include:

train.csv, validation.csv, test.csv

EEG data files in MI/SSVEP folders

2. Run the Code
You can execute this as a Jupyter Notebook or a Python script.

python your_main_script.py
3. Execution Steps
Initialize: Import libraries, set up paths.

Load Metadata: Read train.csv, validation.csv, and test.csv.

Define Functions:

load_trial_data, preprocess_eeg

Feature extraction: PSD, Time-Domain, Connectivity, CSP

## Prepare Datasets:

Apply preprocessing + feature extraction on the train/validation sets

Split by Task:

Separate MI and SSVEP data

### Train Models:

Fit RandomForestClassifier (MI), XGBClassifier (SSVEP)

Evaluate Accuracy

Generate Predictions on the test set



#  🧠 Future Improvements
Integrate deep learning models like EEGNet for end-to-end learning [3].

Implement cross-validation instead of a static train/validation split.

Add artifact removal (e.g., ICA).

Perform feature importance analysis (e.g., SHAP values).

Explore transfer learning to adapt across subjects or sessions.

