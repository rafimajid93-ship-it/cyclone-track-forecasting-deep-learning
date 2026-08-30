# Deep Learning-Based Cyclone Track Forecasting 🌪️

[![Paper](https://img.shields.io/badge/Springer-Published-blue)](https://doi.org/10.1007/978-3-031-53274-0_1)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--53274--0__1-blue)](https://doi.org/10.1007/978-3-031-53274-0_1)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)

Official code repository associated with the published paper:

> **A Deep Learning-Based Study for Cyclone Track Forecasting: Comparative Analysis Using Historical Data from the Bay of Bengal**

**Rafi Majid, Akmam Hasan, Shayrey Mostarin, Kazi Rabiul Alam, and Rashedur M. Rahman**

Published in **Networking and Parallel/Distributed Computing Systems**, *Studies in Computational Intelligence*, Vol. 1125, Springer, 2024, pp. 1–15.

📄 **Paper:** https://doi.org/10.1007/978-3-031-53274-0_1

---

## Overview

Tropical cyclone track forecasting is important for disaster preparedness, especially for highly vulnerable coastal regions around the **Bay of Bengal**.

This research investigates the use of recurrent deep-learning architectures for predicting future cyclone positions from historical cyclone-track data. The published study compares:

- **Long Short-Term Memory (LSTM)**
- **Convolutional LSTM (ConvLSTM / C-LSTM)**
- **Gated Recurrent Unit (GRU)**

The models are trained using historical cyclone information such as latitude, longitude, wind speed, and temporal information, and are evaluated using both conventional regression metrics and geographical track error.

---

## Research Objectives

The study focuses on:

- predicting future cyclone latitude and longitude;
- comparing recurrent deep-learning architectures for cyclone trajectory forecasting;
- evaluating forecasting performance on historical Bay of Bengal cyclone data;
- studying both multi-cyclone historical learning and short time-series forecasting;
- measuring geographical prediction error in kilometers in addition to standard machine-learning metrics.

---

## Methodology

The overall research workflow is:

```text
Historical Cyclone Data
         │
         ▼
Data Cleaning & Preprocessing
         │
         ├── Missing-value analysis
         ├── Outlier detection
         ├── Feature scaling
         └── Cyclone-wise grouping
         │
         ▼
Sequence Construction
         │
         ▼
Deep Learning Models
   ┌─────┼─────────┐
   ▼     ▼         ▼
 LSTM  ConvLSTM   GRU
   └─────┼─────────┘
         ▼
Trajectory Prediction
         │
         ▼
Evaluation
   ├── R²
   ├── MSE
   ├── MAE
   ├── RMSE
   └── Average Distance Error
         │
         ▼
Actual vs. Predicted Track Visualization
```

---

## Dataset

The paper uses historical tropical cyclone data from the **North Indian Ocean / Bay of Bengal region**.

The reported dataset contains:

- **178 cyclones**
- **3,579 observations**
- observations recorded approximately every **6 hours**

Important attributes discussed in the paper include:

```text
Storm Number
Latitude
Longitude
Maximum Wind Speed
Hours From First Observation
```

### Dataset expected by the current notebook

The uploaded notebook expects a CSV file named:

```text
main - main.csv
```

and directly uses the following columns:

```text
Storm Number
x
y
Max Wind Speed (kt)
```

where `x` and `y` represent cyclone coordinates.

> The dataset is not included in this repository unless its redistribution terms permit it. Place your local CSV beside the notebook or upload it when prompted in Google Colab.

---

## Current Code Implementation

The current `Final.ipynb` provides an **LSTM-based cyclone trajectory forecasting implementation**.

### Input features

```python
input_features = [
    "x",
    "y",
    "Max Wind Speed (kt)"
]
```

### Prediction targets

```python
output_features = [
    "x",
    "y"
]
```

### Model

```python
model = Sequential()
model.add(LSTM(
    100,
    activation="relu",
    input_shape=(2, 3)
))
model.add(Dense(2))
```

The model is trained with:

```python
optimizer = "adam"
loss = "mse"
epochs = 100
batch_size = 32
```

The implementation constructs sequences from consecutive cyclone observations and predicts the next latitude/longitude pair.

---

## Important Reproducibility Note

The **published paper contains a broader experimental study than the current notebook**.

The paper evaluates **LSTM, ConvLSTM, and GRU** under multiple experimental settings. The current uploaded notebook contains the **LSTM implementation** and uses this split:

```python
train_data = data[data["Storm Number"] <= 100]
test_data  = data[data["Storm Number"] > 100]
```

The published paper reports experiments including:

1. training on up to **140 historical cyclones** and forecasting later cyclones;
2. forecasting from the **first 60 hours** of cyclone observations;
3. comparisons among **LSTM, ConvLSTM, and GRU**.

Therefore, this repository should currently be treated as the **available LSTM code associated with the study**, rather than a complete reproduction package for every experiment reported in the publication.

---

## Results Reported in the Paper

The following table summarizes the paper's reported **R² score** and **average geographical distance error**.

| Experiment | Model | R² | Distance Error (km) |
|---|---|---:|---:|
| Train ≤140 storms → Cyclone Roanu | LSTM | 0.955 | 65.111 |
|  | C-LSTM | 0.919 | 127.912 |
|  | GRU | **0.974** | 66.050 |
| Train ≤140 storms → next 38 storms | LSTM | 0.741 | 101.144 |
|  | C-LSTM | 0.709 | 108.408 |
|  | GRU | **0.762** | **82.787** |
| First 60 hours → Cyclone Roanu | LSTM | **0.875** | 154.747 |
|  | C-LSTM | 0.405 | **54.595** |
|  | GRU | 0.810 | 183.033 |
| First 60 hours → next 38 storms | LSTM | 0.594 | 191.412 |
|  | C-LSTM | **0.610** | 262.481 |
|  | GRU | 0.603 | **182.938** |

The paper concludes that **GRU provided the strongest overall performance across the comparative study**, while model behavior varied depending on the forecasting setup and evaluation metric.

---

## Evaluation Metrics

The published research evaluates predictions using:

### Mean Squared Error

```text
MSE = mean((actual - predicted)²)
```

### Root Mean Squared Error

```text
RMSE = √MSE
```

### Mean Absolute Error

```text
MAE = mean(|actual - predicted|)
```

### R² Score

Measures how much of the variation in the target coordinates is explained by the model.

### Average Distance Error

Because small latitude/longitude errors can correspond to large geographical distances, the study also calculates the distance between predicted and actual cyclone positions in **kilometers** using the **Haversine formula**.

This provides a more practical measure of cyclone-track forecasting accuracy.

---

## Track Visualization

The notebook uses **Cartopy** and Matplotlib to plot actual and predicted cyclone tracks geographically.

```python
ax.plot(
    y_test[:, 0],
    y_test[:, 1],
    label="Actual"
)

ax.plot(
    y_pred[:, 0],
    y_pred[:, 1],
    label="Predicted"
)
```

This allows visual comparison between the observed cyclone path and the model-generated trajectory.

---

## Installation

### Option 1 — Google Colab

The notebook was written for **Google Colab**.

Open the notebook in Colab and install Cartopy if required:

```bash
pip install cartopy
```

Most other dependencies are normally available in Colab.

### Option 2 — Local environment

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Install the dependencies:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib cartopy
```

Then launch Jupyter:

```bash
jupyter notebook
```

---

## Dependencies

Main libraries used by the current implementation:

```text
Python
TensorFlow / Keras
NumPy
pandas
scikit-learn
Matplotlib
Cartopy
Google Colab
```

---

## How to Run

### 1. Prepare the dataset

Make sure your CSV contains:

```text
Storm Number
x
y
Max Wind Speed (kt)
```

For the current notebook, use the filename:

```text
main - main.csv
```

### 2. Open the notebook

```text
Final.ipynb
```

For a cleaner public repository, consider renaming it to:

```text
cyclone_track_forecasting_lstm.ipynb
```

### 3. Upload the CSV

When running in Colab, the first cell opens a file-upload dialog.

Upload:

```text
main - main.csv
```

### 4. Run all cells

The notebook will:

1. load the cyclone dataset;
2. split training and testing data;
3. normalize the selected features;
4. create time-series sequences;
5. train the LSTM;
6. predict cyclone coordinates;
7. plot actual vs. predicted values;
8. visualize cyclone tracks.

---

## Suggested Repository Structure

```text
cyclone-track-forecasting/
│
├── README.md
├── cyclone_track_forecasting_lstm.ipynb
├── requirements.txt
│
└── data/
    └── README.md
```

Avoid committing large or restricted datasets unless redistribution is explicitly permitted.

For the published paper, linking to the DOI is usually preferable to uploading the publisher PDF unless you have permission to redistribute that version.

---

## Published Paper

**R. Majid, A. Hasan, S. Mostarin, K. R. Alam, and R. M. Rahman**,  
*"A Deep Learning-Based Study for Cyclone Track Forecasting: Comparative Analysis Using Historical Data from the Bay of Bengal,"*  
in **Networking and Parallel/Distributed Computing Systems**, Studies in Computational Intelligence, vol. 1125, Springer, 2024, pp. 1–15.

**DOI:** https://doi.org/10.1007/978-3-031-53274-0_1

---

## Citation

If you use this work, please cite the published paper:

```bibtex
@incollection{majid2024cyclone,
  author    = {Majid, Rafi and
               Hasan, Akmam and
               Mostarin, Shayrey and
               Alam, Kazi Rabiul and
               Rahman, Rashedur M.},
  title     = {A Deep Learning-Based Study for Cyclone Track Forecasting:
               Comparative Analysis Using Historical Data from the Bay of Bengal},
  booktitle = {Networking and Parallel/Distributed Computing Systems},
  series    = {Studies in Computational Intelligence},
  volume    = {1125},
  pages     = {1--15},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-53274-0_1}
}
```

---

## Disclaimer

This repository is intended for **research and educational purposes**.

Cyclone forecasting is a safety-critical task. Predictions produced by this experimental deep-learning implementation should **not** be used as a replacement for forecasts or warnings issued by official meteorological agencies.

---

## Authors

- **Rafi Majid**
- **Akmam Hasan**
- **Shayrey Mostarin**
- **Kazi Rabiul Alam**
- **Rashedur M. Rahman**

Department of Electrical and Computer Engineering,  
North South University, Dhaka, Bangladesh.

---

## Acknowledgment

If this repository helps your research, please consider citing the published Springer chapter.
