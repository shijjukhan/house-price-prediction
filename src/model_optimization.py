from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def optimize_model(X_train, y_train, X_test, y_test):
    # Example with Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f'Ridge Regression MSE: {mse_ridge}')
    print(f'Ridge Regression R^2 Score: {r2_ridge}')

