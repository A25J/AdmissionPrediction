import os
import warnings
import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pickle

warnings.filterwarnings("ignore")

# =========================
# Controllers
# =========================
saveModel = True

# =========================
# Paths and Dataset
# =========================
scriptFolder = os.path.dirname(os.path.abspath(__file__))
csvFile = "Admission_cleaned.csv"
csvFilePath = scriptFolder + "/" + csvFile
RegressorFile = "Predict.sav"
RegressorFilePath = scriptFolder + "/" + RegressorFile

df = pd.read_csv(csvFilePath)

dataset = df.values
X = dataset[:, 0:7]
y = dataset[:, 7]

# =========================
# Train-Test split
# =========================
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# XGB Fine-Tuning
# =========================
print(Fore.CYAN + "\n Tuning XGBRegressor..." + Style.RESET_ALL)

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=3,
                        scoring='neg_root_mean_squared_error', n_jobs=-1)

xgb_grid.fit(XTrain, yTrain)
best_xgb = xgb_grid.best_estimator_

print(Fore.GREEN + " Best XGB Parameters:", xgb_grid.best_params_, Style.RESET_ALL)

# =========================
# Random Forest Fine-Tuning
# =========================
print(Fore.CYAN + "\n Tuning Random Forest..." + Style.RESET_ALL)

rf = RandomForestRegressor(random_state=42)

rf_param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [5, 8, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(rf, rf_param_grid, cv=3,
                       scoring='neg_root_mean_squared_error', n_jobs=-1)

rf_grid.fit(XTrain, yTrain)
best_rf = rf_grid.best_estimator_

print(Fore.GREEN + " Best RF Parameters:", rf_grid.best_params_, Style.RESET_ALL)

# =========================
# CART Fine-Tuning
# =========================
print(Fore.CYAN + "\n Tuning Decision Tree (CART)..." + Style.RESET_ALL)

cart = DecisionTreeRegressor(random_state=42)

cart_param_grid = {
    'max_depth': [3, 5, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

cart_grid = GridSearchCV(cart, cart_param_grid, cv=3,
                         scoring='neg_root_mean_squared_error', n_jobs=-1)

cart_grid.fit(XTrain, yTrain)
best_cart = cart_grid.best_estimator_

print(Fore.GREEN + " Best CART Parameters:", cart_grid.best_params_, Style.RESET_ALL)

# =========================
# SVM Fine-Tuning
# =========================
print(Fore.CYAN + "\n Tuning Support Vector Machine..." + Style.RESET_ALL)

svm = SVR()

svm_param_grid = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(svm, svm_param_grid, cv=3,
                        scoring='neg_root_mean_squared_error', n_jobs=-1)

svm_grid.fit(XTrain, yTrain)
best_svm = svm_grid.best_estimator_

print(Fore.GREEN + " Best SVM Parameters:", svm_grid.best_params_, Style.RESET_ALL)
knn = KNeighborsRegressor()

param_grid = {
    "n_neighbors": [3,5,7,9,11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

print("\nTuning KNN hyperparameters...")
grid_search.fit(XTrain, yTrain)

best_knn = grid_search.best_estimator_

print("Best KNN Parameters:", grid_search.best_params_)
# -------------------------------------------------
# MLR Fine-Tuning using GridSearchCV
# -------------------------------------------------

mlr_models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso()
}

mlr_param_grid = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1]
}

best_mlr = None
best_score = float("inf")
best_name = ""

print("\nTuning MLR models...")

for name, model in mlr_models.items():

    if name == "Linear":
        model.fit(XTrain, yTrain)
        best_mlr = model
        best_name = "Linear Regression"

    else:
        grid = GridSearchCV(
            estimator=model,
            param_grid=mlr_param_grid,
            cv=kfold,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        grid.fit(XTrain, yTrain)

        score = -grid.best_score_

        if score < best_score:
            best_score = score
            best_mlr = grid.best_estimator_
            best_name = name

print("Best MLR Model:", best_name)
# =========================
# Regressors list
# =========================
regressors = [
    ("SVM", "Support Vector Machine (Tuned)", best_svm),
    ("RF", "Random Forest (Tuned)", best_rf),
    ("XGB", "Extreme Gradient Boosting (Tuned)", best_xgb),
    ("CART", "Decision Tree (Tuned)", best_cart),
    ("KNN", "K-Nearest Neighbour (Tuned)", best_knn),
    ("MLR", "Multiple Linear Regression (Tuned) ", best_mlr),
    ("MLP", "Multi-Layer Perceptron", MLPRegressor(
        hidden_layer_sizes=(16, 4),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        batch_size=10,
        random_state=42))
]

# =========================
# Evaluation
# =========================
results = []

kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

for code, name, regressor in regressors:
    scores = cross_val_score(regressor, XTrain, yTrain, cv=kfold, n_jobs=-1,
                             scoring="neg_root_mean_squared_error")

    score = cross_val_score(regressor, XTrain, yTrain, cv=kfold, n_jobs=-1,
                            scoring="r2")

    s = cross_val_score(regressor, XTrain, yTrain, cv=kfold, n_jobs=-1,
                        scoring="neg_mean_absolute_error")

    rmse = -scores.mean()
    mae = -s.mean()
    r2 = score.mean()
    std = scores.std()

    print(name)

    results.append((code, name, regressor,
                    round(rmse, 2),
                    round(std, 2),
                    round(r2, 2),
                    round(mae, 2)))

# =========================
# Sort results
# =========================
results.sort(key=lambda i: i[5], reverse=True)

# =========================
# Display results
# =========================
p1 = round((len(XTrain) * 100) / len(X))
p2 = round((len(XTest) * 100) / len(X))

print("\n Regression Predictive Model: Admission Acceptance")
print(" ---------------------------------------------------")
print("   * Data Set Size:", len(X))
print("\t Training:", len(XTrain), "(" + str(p1) + "%)")
print("\t Test:", len(XTest), "(" + str(p2) + "%)")


# Table header
print("\n{:<3} {:<40} {:<10} {:<10} {:<10} {:<10}".format(
    "#", "Regressor", "RMSE", "Std Dev", "R2", "MAE"))
print("-" * 90)

# Table rows
i = 1
for r in results:
    name = r[0] + " - " + r[1]
    print("{:<3} {:<40} {:<10} {:<10} {:<10} {:<10}".format(
        i,
        name,
        r[3],
        r[4],
        r[5],
        r[6]
    ))
    i += 1

# =========================
# Save best model
# =========================
if saveModel:
    bestRegressor = results[0][2]
    bestRegressor.fit(X, y)

    print(Fore.MAGENTA,
          "\n Best Regressor:", results[0][0], "-", results[0][1])
    print(Style.RESET_ALL)

    pickle.dump(bestRegressor, open(RegressorFilePath, "wb"))

    print(Fore.GREEN + "\n Final Regressor Saved As: '" + RegressorFile + "'",
          Fore.RESET)