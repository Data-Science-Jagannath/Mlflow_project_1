import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

exp_id = mlflow.create_experiment('loan_prediction')

with mlflow.start_run(run_name = 'xgboost') as run:
    pass
mlflow.end_run()

n_estimators = 10
criterion = 'gini'

mlflow.log_param('n_estimators':n_estimators)
mlflow.log_param('criterion':criterion)

