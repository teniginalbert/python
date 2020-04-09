import mglearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)

print('lr.coef_: {}'.format(lr.coef_))
print('lr.intercept_: {}'.format(lr.intercept_))

print('Правильность на обучающем наборе: {:.2f}'.format(lr.score(X_train, y_train)))
print('Правильность на тестовом наборе: {:.2f}'.format(lr.score(X_test, y_test)))