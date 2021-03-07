import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = load_boston()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['MEDV'] = data.target
X = df[['RM']]
y = df['MEDV']

model = LinearRegression()
X_test, X_train, y_test, y_train = train_test_split(X, y, random_state=0)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

plt.figure()
plt.scatter(X_test, y_test, c='red', label='Testing data')
plt.plot(X_test, y_predict, label='Predicted data', linewidth=3)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.legend(loc='upper left')
plt.show()

new_RM = np.array([6.5]).reshape(-1,1)
model.predict(new_RM)
print('$', float(model.predict(new_RM)))

residuals = y_test - y_predict

plt.figure()
plt.scatter(X_test, residuals)
plt.hlines(y=0, xmin=X_test.min(), xmax=X_test.max(), linestyle='--')
plt.xlim((4,9))
plt.xlabel('RM')
plt.ylabel('residuals')
plt.show()

mean_squared_error(y_test, y_predict)
