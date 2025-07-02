import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc", auc)
    mlflow.sklearn.log_model(model, "model")
    mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/model", "SaaS-Churn")

