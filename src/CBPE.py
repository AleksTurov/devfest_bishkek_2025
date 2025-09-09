import nannyml as nml
import pandas as pd
from IPython.display import display

# Load real-world data:
reference_df, analysis_df, _ = nml.load_us_census_ma_employment_data()
display(reference_df.head())
display(analysis_df.head())

# Choose a chunker or set a chunk size:
chunk_size = 5000

# initialize, specify required data columns, fit estimator and estimate:
estimator = nml.CBPE(
    problem_type='classification_binary',
    y_pred_proba='predicted_probability',
    y_pred='prediction',
    y_true='employed',
    metrics=['roc_auc'],
    chunk_size=chunk_size,
)
estimator = estimator.fit(reference_df)
estimated_performance = estimator.estimate(analysis_df)

# Show results:
figure = estimated_performance.plot()
figure.show()

# Define feature columns:
features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
       'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P']

# Initialize the object that will perform the Univariate Drift calculations:
univariate_calculator = nml.UnivariateDriftCalculator(
    column_names=features,
    chunk_size=chunk_size
)

univariate_calculator.fit(reference_df)
univariate_drift = univariate_calculator.calculate(analysis_df)

# Get features that drift the most with count-based ranker:
alert_count_ranker = nml.AlertCountRanker()
alert_count_ranked_features = alert_count_ranker.rank(univariate_drift)
display(alert_count_ranked_features.head())

# Plot drift results for top 3 features:
figure = univariate_drift.filter(column_names=['RELP','AGEP', 'SCHL']).plot()
figure.show()

# Compare drift of a selected feature with estimated performance
uni_drift_AGEP_analysis = univariate_drift.filter(column_names=['AGEP'], period='analysis')
figure = estimated_performance.compare(uni_drift_AGEP_analysis).plot()
figure.show()

# Plot distribution changes of the selected features:
figure = univariate_drift.filter(period='analysis', column_names=['RELP','AGEP', 'SCHL']).plot(kind='distribution')
figure.show()

# Get target data, calculate, plot and compare realized performance with estimated performance:
_, _, analysis_targets_df = nml.load_us_census_ma_employment_data()

analysis_with_targets_df = pd.concat([analysis_df, analysis_targets_df], axis=1)
display(analysis_with_targets_df.head())

performance_calculator = nml.PerformanceCalculator(
    problem_type='classification_binary',
    y_pred_proba='predicted_probability',
    y_pred='prediction',
    y_true='employed',
    metrics=['roc_auc'],
    chunk_size=chunk_size)

performance_calculator.fit(reference_df)
calculated_performance = performance_calculator.calculate(analysis_with_targets_df)

figure = estimated_performance.filter(period='analysis').compare(calculated_performance).plot()
figure.show()
