import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DATA
df = pd.read_csv("house_price.csv")


X = df[['bedroom', 'size']]
y = df['price']

# Test size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)



sgd = SGDRegressor(max_iter=1000)
sgd.fit(X_train_scaled, y_train)
y_pred_sgd = sgd.predict(X_test_scaled)

# EVALUATE

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, rmse, mape


lr_mae, lr_mse, lr_rmse, lr_mape = evaluate_model(y_test, y_pred_lr)


sgd_mae, sgd_mse, sgd_rmse, sgd_mape = evaluate_model(y_test, y_pred_sgd)


print("LinearRegression Coefficients:", lr.coef_)
print("SGDRegressor Coefficients:", sgd.coef_)


results = pd.DataFrame({
    "Model": ["LinearRegression", "SGDRegressor"],
    "MAE": [lr_mae, sgd_mae],
    "MSE": [lr_mse, sgd_mse],
    "RMSE": [lr_rmse, sgd_rmse],
    "MAPE": [lr_mape, sgd_mape]
})


print("\nResult:")
print(results)

print("Explain the trade-offs between these metrics.")
print("\n")
print("MAE(Mean Absolute Error):")
print("This will take the average of the all absolute errors, no negative signs, only the size of the mistake")
print("It's east to understand and use the same units for your target")
print("\n")
print("MSE(Mean Squared Error):")
print("The squares all the errors before averaging, therefore they will have bigger mistakes count")
print("Indicates the big mistakes more. If you want to avoid huge errors, MSE will point it out. ")
print("\n")
print("RMSE(Root Mean Squared Error):")
print("Consider big errors, but the result s in the original units, so it's easier to interpret.")
print("Same as MSE, but it's more sensitive to outliers, but bad prediction can it look much worse.")
print("\n")
print("MAPE(Mean Absolute Percentage Error):")
print("MAPE will provide the average percentage error, it shows error as a percentage, easy to explain and comparing different models.")
