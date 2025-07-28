import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# DATA
train_df = pd.read_csv('mnist_train.csv')
test_df  = pd.read_csv('mnist_test.csv')

X_train, y_train = train_df.iloc[:, 1:].values, train_df.iloc[:, 0].values
X_test,  y_test  = test_df.iloc[:, 1:].values,  test_df.iloc[:, 0].values


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_std, y_train)
y_pred_lr = lr.predict(X_test_std)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression accuracy: {acc_lr:.4f}")


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std, y_train)
y_pred_knn = knn.predict(X_test_std)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN accuracy: {acc_knn:.4f}") 