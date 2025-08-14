# TUNIndex Pro

TUNIndex Pro is a web application for analyzing and forecasting the TUNINDEX (Tunisian Stock Market Index) using various time series models.

## Project Structure

TUNIndex-Pro/
├── app.py                  # Main Flask application
├── uploads/                # Folder for uploaded CSV files (created automatically)
├── static/                 # Static files (CSS, JS)
│   ├── css/
│   │   └── style.css       # Custom stylesheets
│   └── js/
│       └── script.js       # Custom JavaScript
├── templates/              # HTML templates
│   └── index.html          # Main page template
├── requirements.txt        # Python dependencies
└── README.md               # This file

## Features

* **CSV Data Upload**: Upload historical TUNINDEX data.
* **Data Overview**: View descriptive statistics and plots of the closing price.
* **Time Series Analysis**:
    * Stationarity tests (ADF, KPSS).
    * Correlation analysis with heatmap.
* **Predictive Modeling**:
    * Train models: ETS, ARIMA, SVR, XGBoost, MLP.
    * Hyperparameter tuning for ML models using GridSearchCV.
* **Model Evaluation**:
    * Compare model performance using metrics (RMSE, MAE, MAPE, R2).
    * Visualize actual vs. predicted values.
* **Future Predictions**: Generate future price predictions using trained models.
* **Data Export**: Export analysis results and predictions to CSV or Excel.
* **User Interface**:
    * Clear, functional design with sidebar navigation.
    * Interactive (display of) graphs.
    * Dark mode option.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd TUNIndex-Pro
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    python app.py
    ```

5.  Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Python Version
This project is developed and tested with Python 3.10.

## Usage

1.  **Upload Data**: Start by uploading a CSV file containing TUNINDEX historical data. The CSV should ideally have columns like 'Date', 'Close', 'Open', 'High', 'Low', 'Vol.', 'Change %'.
2.  **Data Overview**: Check the basic statistics and the trend of the closing prices.
3.  **Analysis**: Perform stationarity and correlation analyses.
4.  **Modeling**: Train the available forecasting models. View their training summaries and initial metrics.
5.  **Evaluate**: Compare the performance of different models on train and test sets.
6.  **Predict**: Select a trained model and specify the number of future steps to predict.
7.  **Export**: Download the original data, along with any generated predictions.

## Note on Real-time Updates

The current version does not support real-time data feeds and updates. This would require a more complex setup involving WebSockets or Server-Sent Events (SSE) and a live data source for TUNINDEX. This can be considered a future enhancement.
