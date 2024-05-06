import mlflow
logged_model = 'runs:/d3219778ac644105b99732403c15255e/RandomForestClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                3.0,
                1.0,
                2.0,
                8.698
            ]]
print(f"Prediction is: {loaded_model.predict(pd.DataFrame(data))}")