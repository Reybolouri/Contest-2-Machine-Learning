import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
train_url = 'https://raw.githubusercontent.com/Reybolouri/Contest-2-Machine-Learning/main/data/train.csv'
test_url  = 'https://raw.githubusercontent.com/Reybolouri/Contest-2-Machine-Learning/main/data/test.csv'

train_df = pd.read_csv(train_url)
test_df  = pd.read_csv(test_url)

# 2. Features and (zero-based) labels
X       = train_df.drop(['id','y'], axis=1).values
y       = train_df['y'].values.astype(int) - 1    # <-- shift labels to {0,1,2}
X_test  = test_df.drop('id', axis=1).values

# 3. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 4. Random Forest baseline
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_val_preds = rf.predict(X_val)
print("RF Validation Accuracy:", accuracy_score(y_val, rf_val_preds))

# 5. XGBoost with GridSearchCV
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [3, 5]
}
grid = GridSearchCV(
    xgb,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    error_score='raise'    # will now raise if anything still goes wrong
)
grid.fit(X_train, y_train)

print("XGB Best Params:", grid.best_params_)
print("XGB CV Accuracy:", grid.best_score_)

# 6. Retrain best XGB on all data
best_non_nn = grid.best_estimator_
best_non_nn.fit(X, y)

# 7. Predict on test set and shift labels back to {1,2,3}
test_preds_zero_based = best_non_nn.predict(X_test)
test_preds = test_preds_zero_based + 1

# 8. Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'y':  test_preds
})
submission.to_csv('non_nn.csv', index=False)
print("Saved → submission_non_nn.csv")
