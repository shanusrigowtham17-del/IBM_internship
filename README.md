# Advanced Real Estate Valuation Engine

What we have deployed here is an enterprise-grade machine learning pipeline for real estate price prediction. We are utilizing battle-tested algorithms—Scikit-Learn, XGBoost, and LightGBM—hooked into a Streamlit frontend for immediate stakeholder accessibility. It’s robust, it handles the data heavy lifting in the background, and it gets the job done without unnecessary overhead.

## Prerequisites

Before you spin this up, ensure your infrastructure meets the following baseline:
* Python 3.8 or higher installed on your system.
* A stable network connection to pull down the required binaries.
* The `data.csv` payload securely staged in your execution directory.

## Environment Provisioning 

Run clean environments to avoid polluting the global scope. Spin up your virtual environment, then execute the standard deployment protocol:

```bash
pip install -r requirements.txt
```

## System Execution

Once your dependencies are resolved and the data is mounted in the correct working directory, initialize the application server:

```bash
streamlit run app.py
```
    
The service will bind to your local port. Open your browser, review the UI, and start running your inference models. The system will cache the initial training cycle in memory to optimize subsequent rendering speeds.

## Maintenance and Troubleshooting

* **FileNotFoundError:** If the server throws this, it means you didn't stage `data.csv` correctly. Check your absolute paths. 
* **Metric Drift:** Model metrics (MAE, RMSE, R-Squared) are calculated in real-time. If those numbers start drifting on new data payloads, it's time to retrain or check for data contamination.
