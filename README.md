# automl-market-regime-stock-prediction
An AutoML-based financial analytics framework integrating K-Means clustering for market regime detection, FLAML-driven model optimisation, and supervised stock movement prediction. Features risk analytics, technical indicator engineering, and backtesting for risk-aware investment decision support.
Design Auto ML for Market Regime Detection and Risk-Adjusted Stock Movement Prediction

An Automated Machine Learning Framework for Financial Time-Series Analysis, Market Regime Identification, Directional Forecasting, and Risk-Aware Decision Support

1. Project Overview
Financial markets are inherently dynamic and exhibit non-stationary behaviour, where statistical properties such as returns, volatility, momentum, and market correlations change over time.
Traditional predictive models often assume relatively stable relationships between input variables and target outcomes. However, financial markets frequently transition between different behavioural conditions, including bullish, bearish, sideways, and high-volatility environments. These transitions can affect the predictive reliability of conventional forecasting approaches.
This project presents an integrated Automated Machine Learning (AutoML) framework designed to identify market regimes, predict short-term stock price direction, and evaluate potential investment decisions through risk-adjusted performance analysis.
The framework combines financial feature engineering, unsupervised machine learning, automated model selection, predictive classification, risk assessment, and historical backtesting into a unified analytical pipeline.
The initial implementation focuses on Apple Inc. (AAPL) historical market data and provides an interactive interface through Streamlit.
The objective is not simply to forecast stock prices, but to investigate how market conditions and associated risk characteristics can be incorporated into a systematic stock-movement prediction framework.


2. Project Objectives
The primary objectives of this project are:

1. Market Regime Identification: Identify distinct market conditions using unsupervised machine learning techniques.
2. Financial Feature Engineering: Extract meaningful technical and statistical indicators from historical OHLCV data.
3. Automated Model Selection: Apply AutoML to automate the selection and optimisation of suitable predictive models.
4. Stock Movement Prediction: Formulate the forecasting problem as a supervised classification task to estimate the direction of subsequent stock-price movement.
5. Risk Assessment: Evaluate volatility and other risk-related characteristics to contextualise model predictions.
6. Decision Support: Translate predictive outputs into interpretable BUY, HOLD, and AVOID signals.
7. Performance Evaluation: Assess the predictive strategy using historical backtesting and financial performance metrics.
8. Interactive Visualisation: Develop a Streamlit-based dashboard for displaying market analysis and prediction results.
   

3. System Architecture
The framework follows a sequential analytical pipeline:

          Historical Market Data
                   |
                   v
          Data Acquisition
                   |
                   v
        Data Cleaning and Validation
                   |
                   v
          Feature Engineering
                   |
                   v
          Market Regime Detection
                   |
                   v
             AutoML Engine
                   |
                   v
        Model Selection and Tuning
                   |
                   v
       Stock Direction Prediction
                   |
                   v
             Risk Analysis
                   |
                   v
       BUY / HOLD / AVOID Signals
                   |
                   v
          Historical Backtesting
                   |
                   v
         Performance Evaluation
                   |
                   v
        Streamlit Dashboard

Functional Components

Component	Description
Data Acquisition	retrieves historical stock-market information.
Data Preprocessing	Organizes and validates financial observations.
Feature Engineering	generates technical indicators and statistical features.
Regime Detection	uses K-Means clustering to identify market-state groupings.
AutoML	Automates predictive model selection and hyperparameter optimization.
Classification	Predicts the direction of subsequent stock-price movement.
Risk Analysis	evaluates volatility and risk-related characteristics.
Decision Engine	generates BUY, HOLD, or AVOID recommendations.
Backtesting	evaluates the strategy against historical market observations.
Dashboard	presents analytical results through an interactive interface.


4. Methodology

4.1 Historical Financial Data Acquisition
Historical stock-market data is obtained using the Yahoo Finance data interface through the yfinance Python library.

The initial experimental dataset focuses on Apple Inc. (AAPL).

The dataset contains:

* Date
* Open Price
* High Price
* Low Price
* Closing Price
* Trading Volume
* Dividends
* Stock Splits

These observations provide the foundation for subsequent statistical analysis and feature construction.

4.2 Data Preprocessing
The acquired data undergoes preprocessing to ensure consistency and suitability for machine learning.

The preprocessing stage involves:

* Date formatting and chronological organization.
* Handling missing observations.
* Validation of numerical attributes.
* Preparation of the dataset for feature engineering.
* Construction of time-dependent target variables.

Chronological ordering is essential because financial observations are temporally dependent.

4.3 Financial Feature Engineering
Feature engineering transforms raw market observations into variables that represent different aspects of market behaviour.

The implemented features include:

Feature	Description
Daily Return	Percentage change in closing price.
MA20	20-period moving average.
MA50	50-period moving average.
Volatility	Rolling standard deviation of returns.
Momentum	measures recent price movement.
MA20 Ratio: The relative relationship between the closing price and the MA20.
MA50 Ratio: The relative relationship between the closing price and the MA50.
RSI	Relative Strength Index, used to examine momentum conditions.
Volume, Change	in trading volume.
Volume MA20 Ratio: Relative trading-volume activity.

These variables provide information about price trends, momentum, volatility, and trading activity.

The engineered features are subsequently used in market-regime identification and predictive modelling.

4.4 Market Regime Detection
Market regimes represent different behavioural conditions observed in financial markets.

This project employs K-Means clustering, an unsupervised learning algorithm, to identify groups of observations with similar feature characteristics.

The clustering process involves:

1. Selecting relevant market features.
2. Scaling the input variables.
3. Applying K-Means clustering.
4. Assigning observations to their respective clusters.
5. Examining cluster characteristics to interpret market conditions.

The identified clusters are used to distinguish market environments that may exhibit different combinations of trend, momentum, and volatility.

Important: K-Means generates numerical cluster labels rather than inherently identifying bullish or bearish markets. Economic interpretations must be assigned through analysis of the resulting cluster characteristics.

4.5 Automated Machine Learning
The predictive modelling component employs FLAML (Fast and Lightweight AutoML).
AutoML automates important aspects of the machine learning workflow, including:

* Model selection.
* Hyperparameter optimisation.
* Evaluation of candidate models.
* Identification of a suitable predictive estimator under the specified optimisation objective and computational budget.

The framework uses automated model selection rather than relying exclusively on manually selected predictive algorithms.

The current experimental implementation identifies LightGBM as the selected model.

LightGBM is a gradient-boosting framework based on decision-tree learning and is designed for efficient training on structured datasets.

4.6 Stock Movement Prediction
The forecasting task is formulated as a supervised binary classification problem.
Instead of directly estimating the future stock price, the model predicts the direction of subsequent price movement.
The target variable represents whether the next-period return is positive or non-positive.
The predictive component estimates the probability associated with the upward movement class.
This probability is subsequently used by the decision-support component.
Prediction horizon: Next trading observation (one-step-ahead directional prediction).

4.7 Risk Analysis and Decision Support
Predictive output alone is insufficient for evaluating a potential trading decision.
The framework, therefore, incorporates risk-related analysis to contextualise the predicted market direction.
The decision-support component generates three possible signals:

Signal	Interpretation
BUY:	Conditions satisfy the implemented criteria for considering a long position.
HOLD: The system does not identify a sufficiently actionable opportunity.
AVOID: The current conditions do not satisfy the implemented criteria for entering a position.

These signals are generated through the project’s implemented prediction and risk-assessment logic.
They are intended as analytical decision-support outputs, not guaranteed investment recommendations.

4.8 Historical Backtesting
Backtesting evaluates the behaviour of a trading strategy using historical observations.
The implemented strategy is compared against a Buy-and-Hold benchmark.
The backtesting implementation incorporates:

* Chronological training and testing separation.
* Predicted class probabilities.
* Long-only position generation.
* Transaction-cost adjustment.
* Portfolio return calculation.
* Risk and performance evaluation.

The current implementation applies a transaction-cost assumption of 0.1%.

Backtesting provides a mechanism for assessing whether predictive outputs translate into meaningful historical strategy performance.


5. Performance Evaluation
The framework evaluates performance using both financial-return metrics and risk-adjusted indicators.

Evaluation Metrics

Metric	Purpose
Cumulative Return	measures the overall percentage gain or loss over the evaluation period.
Volatility	measures the variability in strategy returns.
Sharpe Ratio	evaluates return relative to total volatility.
Sortino Ratio	evaluates return relative to downside volatility.
Maximum Drawdown	Measures the largest peak-to-trough decline in portfolio value.

Experimental Results
The current backtesting experiment produced the following results:

Metric	AutoML Strategy	Buy-and-Hold
Cumulative Return	-2.10%	+12.21%
Volatility	13.40%	28.58%
Sharpe Ratio	-0.371	1.259
Sortino Ratio	-0.219	1.157
Maximum Drawdown	-8.68%	-11.05%

Values are reported from the current implementation’s backtesting output. They should be interpreted using the evaluation period and metric conventions of that experiment.

Interpretation
The current results demonstrate an important distinction between predictive modelling and investment performance.
Although the AutoML framework incorporates automated classification and risk-related decision logic, its current backtested strategy underperformed the Buy-and-Hold benchmark in cumulative return and risk-adjusted performance.
The strategy exhibited lower volatility and a smaller maximum drawdown in the reported experiment.
However, lower volatility alone does not establish superior investment performance.
These findings indicate that further investigation is necessary into model calibration, regime-conditioned predictive performance, trading thresholds, transaction costs, and strategy robustness.
The project therefore, treats backtesting as an essential evaluation mechanism rather than assuming that machine learning predictions automatically produce profitable trading strategies.


6. Technology Stack
The project uses the following technologies:
Technology	Application
Python	Core programming language
Pandas	Data manipulation and analysis
NumPy	Numerical computation
Scikit-learn: Machine learning utilities and preprocessing
FLAML	Automated machine learning
LightGBM	Gradient-boosting classification
Joblib	Model and artifact serialization
yfinance	Historical financial data acquisition
Streamlit	Interactive web application
Jupyter Notebook: Exploratory analysis and experimentation


7. Repository Structure
automl-market-regime-stock-prediction/
│
├── app.py
│
├── live_pipeline.py
│
├── automl_stock_model.pkl
│
├── market_regime_artifacts.pkl
│
├── data/
│   ├── apple_historical_data.csv
│   └── apple_features_and_regimes.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_claude_model.ipynb
│   └── 03_noideawhatthisis.ipynb
│
├── results/
│   ├── automl_backtest_results.csv
│   └── automl_performance_comparison.csv
│
└── .gitignore

File Descriptions
* app.py: Implements the Streamlit application interface.
* live_pipeline.py: Contains the live stock-analysis and prediction pipeline.
* automl_stock_model.pkl: Serialized trained AutoML model.
* market_regime_artifacts.pkl: Stores the fitted market-regime preprocessing and clustering artifacts.
* data/: Contains historical and engineered financial datasets.
* notebooks/: Contains exploratory analysis, modelling experiments, and development notebooks.
* results/: Contains historical backtesting outputs and model-performance comparisons.


8. Installation and Execution
Prerequisites
* Python 3.11 or a compatible Python environment.
* pip package manager.
* Git.
* Internet connectivity for retrieving live market data.

Step 1: Clone the Repository
git clone https://github.com/atishijain247-png/automl-market-regime-stock-prediction.git

Step 2: Navigate to the Project Directory
cd automl-market-regime-stock-prediction

Step 3: Create a Virtual Environment
python -m venv .venv

Activate the environment.
macOS / Linux:

source .venv/bin/activate
Windows:
.venv\Scripts\activate

Step 4: Install Dependencies
pip install numpy pandas scikit-learn joblib streamlit yfinance flaml lightgbm

Step 5: Launch the Application
streamlit run app.py
The application will launch in a local browser window.


10. Limitations
The current implementation has several limitations that should be considered when interpreting its results.

1. Limited Asset Coverage: Initial experimentation focuses on Apple Inc. (AAPL). Generalization across different companies and financial markets has not yet been established.
2. Historical Dependence: Predictive models are trained using historical observations, which may not fully represent future market behaviour.
3. Market Non-Stationarity: Changes in market structure and statistical distributions can reduce predictive reliability.
4. Clustering Interpretation: K-Means cluster assignments require additional interpretation and do not inherently correspond to predefined economic regimes.
5. Backtesting Limitations: Historical strategy performance does not guarantee future performance. Actual execution may involve additional costs, slippage, liquidity constraints, and market impact.
6. Model Generalization: Performance across different market periods and asset classes requires further validation.
7. Risk Estimation: The current risk-assessment framework is not a comprehensive institutional portfolio-risk management system.


11. Future Scope
Several extensions could improve the framework’s analytical depth and practical applicability.
Model Development
* Experimentation with additional classification algorithms.
* Regime-specific predictive models.
* Ensemble learning.
* Improved probability calibration.
* Model explainability using SHAP.

Market Analysis
* Extension to NIFTY 50 and S&P 500 constituents.
* Multi-asset regime identification.
* Incorporation of macroeconomic indicators.
* Financial news and sentiment analysis.
* Cross-asset dependency analysis.

Risk Management
* Dynamic position sizing.
* Advanced volatility forecasting.
* Value at Risk (VaR).
* Conditional Value at Risk (CVaR).
* Transaction-cost sensitivity analysis.

Evaluation
* Walk-forward validation.
* Rolling-window evaluation.
* Statistical significance testing.
* Regime-specific performance analysis.
* Robustness testing across different historical periods.

Deployment
* Expanded interactive dashboard.
* Multi-ticker analysis.
* Automated model monitoring.
* Periodic model retraining.
* Improved real-time data integration.

These extensions would allow the framework to evolve into a more comprehensive regime-aware financial analytics and decision-support system.


12. Selected Academic References

The following literature provides the theoretical and methodological foundation for the concepts explored in this project.
1. Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77–91.
2. Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk. The Journal of Finance, 19(3), 425–442.
3. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. The Journal of Finance, 25(2), 383–417.
4. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. Econometrica, 57(2), 357–384.
5. Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. Econometrica, 50(4), 987–1007.
6. Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal of Econometrics, 31(3), 307–327.
7. MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.
8. Lloyd, S. (1982). Least Squares Quantization in PCM. IEEE Transactions on Information Theory, 28(2), 129–137.
9. Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32.
10. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
11. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems, 30.
12. Feurer, M., et al. (2015). Efficient and Robust Automated Machine Learning. Advances in Neural Information Processing Systems, 28.
13. Wang, C., et al. (2021). FLAML: A Fast and Lightweight AutoML Library. Proceedings of Machine Learning and Systems, 3.
14. Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.
15. Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction. Computational Statistics & Data Analysis, 120, 70–83.
16. Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). The Probability of Backtest Overfitting. Journal of Computational Finance, 20(4).
17. López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
18. Kelly, B. T., & Xiu, D. (2023). Financial Machine Learning. Foundations and Trends in Finance, 13(3–4), 205–363.
19. Suárez-Cetrulo, A. L., Quintana, D., & Cervantes, A. (2023). Machine Learning for Financial Prediction Under Regime Change Using Technical Analysis: A Systematic Review. International Journal of Interactive Multimedia and Artificial Intelligence.
20. Olorunnimbe, K., & Viktor, H. (2023). Deep Learning in the Stock Market—A Systematic Survey of Practice, Backtesting, and Applications. Artificial Intelligence Review, 56, 2057–2109.
  


13. Disclaimer
This project is an academic and experimental implementation developed for educational and research purposes.
The predictions and trading signals generated by the framework should not be interpreted as financial advice or guarantees of investment performance.
Historical backtesting results do not establish future profitability. Financial markets involve substantial uncertainty, and any practical deployment would require additional validation, risk controls, and regulatory considerations.


14. Author
Atishi Jain
Second Year, Electrical and Computer Engineering Student
Madhav Institute of Science and Technology, Gwalior
Project Domain: Automated Machine Learning | Financial Technology | Quantitative Analysis
GitHub Repository: atishijain247-png/automl-market-regime-stock-prediction

This project represents an exploration of the intersection of machine learning, financial markets, statistical analysis, and automated decision-support systems.
