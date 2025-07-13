# AIC-Competition-
EEG Signal Classification for Brain-Computer Interfaces (BCI) This repository presents a solution for classifying Electroencephalography (EEG) signals, specifically designed for Brain-Computer Interface (BCI) applications. Our project focuses on distinguishing between two primary EEG paradigms: Steady-State Visual Evoked Potentials (SSVEP) and Motor Imagery (MI). Our work involves processing multi-channel EEG recordings, extracting relevant features, and training machine learning models to accurately identify visual stimulus frequencies (for SSVEP) and motor imagery categories (for MI). The main objective of this project is to develop robust and accurate AI models for the classification of EEG signals. By effectively classifying these signals, we aim to contribute to the advancement of non-invasive BCI systems, which hold significant potential for communication and control applications. The project utilizes a dataset comprising multi-channel EEG recordings. The data is organized with train.csv, validation.csv, and test.csv files serving as metadata indices. These index files point to individual EEGdata.csv files, each containing raw EEG signals for a specific trial. 







# Methodology
Our solution employs a comprehensive multi-stage pipeline for EEG signal classification:

## 1. Data Loading and Preprocessing
EEG data for each trial is dynamically loaded based on the metadata provided in train.csv, validation.csv, and test.csv.

Bandpass Filtering: A Butterworth bandpass filter (1-40 Hz) is applied to the raw EEG signals. This step is crucial for removing unwanted noise and isolating the frequency components most relevant to BCI paradigms.

Epoching: Although not explicitly shown in the provided preprocess_eeg function, typical EEG processing involves segmenting continuous data into epochs aligned with specific events or time windows of interest. The current implementation assumes the EEGdata.csv files already contain pre-epoched data or that the feature extraction functions handle the time window implicitly.

Baseline Correction: A baseline correction is applied to normalize the signal, reducing the impact of DC offsets and slow drifts.

Channel Selection: A predefined set of EEG channels (eeg_channels) is utilized, focusing on channels relevant for BCI tasks.

## 2. Feature Extraction
We extract a diverse set of features from the preprocessed EEG signals to capture different aspects of brain activity:

Power Spectral Density (PSD) Features:

Calculated using Welch's method, these features quantify the power distribution across various standard EEG frequency bands: Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (12-30 Hz), and Gamma (30-40 Hz).

The average power within each band for each channel is used as a feature, providing insights into oscillatory brain activity.

## Time-Domain Features:

For each EEG channel, we compute statistical measures such as mean, standard deviation, maximum value, minimum value, kurtosis, and skewness. These features capture the amplitude distribution and shape of the EEG waveforms over time.

Connectivity Features:

The Pearson correlation matrix is calculated between all pairs of EEG channels. This matrix reflects the linear relationships between signals from different brain regions.

The unique elements from the upper triangle of this correlation matrix (excluding the diagonal) are extracted as features, representing the functional connectivity between channels.

## Common Spatial Pattern (CSP) Features:

CSP is a powerful spatial filtering technique, particularly effective for MI tasks. It learns spatial filters that maximize the variance of one class while minimizing the variance of another, thereby enhancing the discriminability of different mental states. This technique is specifically applied to MI task data.

## 3. Model Training
Recognizing the distinct characteristics of SSVEP and MI signals, we employ a strategy of training separate classification models for each task:

Motor Imagery (MI) Model: A RandomForestClassifier is used for MI classification. This model is robust and handles high-dimensional data well. A StandardScaler is applied beforehand to normalize the features, which helps in improving model performance.

Steady-State Visual Evoked Potentials (SSVEP) Model: An XGBClassifier is chosen for SSVEP classification. XGBoost is known for its efficiency and strong predictive performance. Similar to the MI model, a StandardScaler is used for feature normalization.

LabelEncoder is utilized to convert the categorical labels (e.g., 'Left', 'Right' for MI; specific frequencies for SSVEP) into numerical formats required by the machine learning models.

## 4. Prediction and Submission
After training, the respective models are used to predict labels for the held-out test set.

The predictions from both the MI and SSVEP models are combined.

Finally, a submission.csv file is generated, containing the ID of each test trial and its corresponding predicted label, formatted as required for submission.

# Evaluation
The primary evaluation metric for this project is the mean classification accuracy over a held-out test set. This metric is computed separately for SSVEP and MI trials and then averaged to ensure a balanced assessment of performance across both paradigms.

# Dependencies
To run this project, you will need the following Python libraries:

numpy

pandas

scipy

matplotlib

seaborn

scikit-learn

mne

xgboost

You can install these dependencies using pip:

pip install numpy pandas scipy matplotlib seaborn scikit-learn mne xgboost

## How to Run the Code
## Dataset Setup:

Ensure that the mtcaic3 dataset is available in the specified BASE_PATH. In a typical Kaggle environment, this path is /kaggle/input/mtcaic3/. If running locally, download the dataset and adjust BASE_PATH accordingly. The dataset should contain train.csv, test.csv, validation.csv, and the SSVEP/MI subdirectories with EEGdata.csv files.

Execute the Python Script/Notebook:
The project code is structured to be run sequentially, typically within a Jupyter Notebook or as a Python script. Execute the cells or script in the following order:

Initial Setup: Load necessary libraries and define the BASE_PATH.

Load Metadata: Read train.csv, validation.csv, test.csv, and sample_submission.csv.

Define Helper Functions: Ensure load_trial_data, preprocess_eeg, extract_psd_features, extract_time_features, extract_connectivity_features, extract_csp_features, and extract_features are defined. These functions encapsulate the data loading, preprocessing, and feature extraction logic.

Prepare Datasets: Call prepare_dataset for train_df and validation_df to process the EEG data and extract features.

Split Data by Task: Separate the prepared data into MI and SSVEP subsets.

Encode Labels: Initialize and fit LabelEncoder for MI and SSVEP labels.

Train Models: Train the RandomForestClassifier for MI and the XGBClassifier for SSVEP.

Evaluate Models: Calculate and print validation accuracies for both tasks and overall.

Prepare Test Data and Predict: Process the test_df to extract features and generate predictions using the trained models.

Create Submission File: Generate submission.csv with the final predictions.

If running as a Python script, you would execute it from your terminal:

python your_main_script.py
