# ----------------------------
# 5. Model Training
# ----------------------------
# src/train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import wandb

wandb.init(project="saas_churn")

# Load data and preprocess
X = df.drop('churn', axis=1)
y = df['churn']
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='object').columns

preprocessor = build_preprocessor(num_cols, cat_cols)
X_processed = preprocessor.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_processed, y)

# Evaluate
f1 = f1_score(y, model.predict(X_processed))
auc = roc_auc_score(y, model.predict_proba(X_processed)[:,1])
wandb.log({"f1": f1, "auc": auc})

# Save model
joblib.dump(model, "models/churn_model.pkl")
wandb.finish()
