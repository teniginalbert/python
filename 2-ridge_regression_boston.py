import mglearn
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
ridge = Ridge().fit(X_train, y_train)
ridge5 = Ridge(alpha=5).fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge05 = Ridge(alpha=0.5).fit(X_train, y_train)
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
ridge001 = Ridge(alpha=0.01).fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)

print('Правильность на обучающем наборе: {:.2f}'.\
      format(ridge.score(X_train, y_train)))
print('Правильность на тестовом наборе: {:.2f}'.\
      format(ridge.score(X_test, y_test)))

print('Правильность на обучающем наборе (alpha=5): {:.2f}'.\
      format(ridge5.score(X_train, y_train)))
print('Правильность на тестовом наборе (alpha=5): {:.2f}'.\
      format(ridge5.score(X_test, y_test)))

print('Правильность на обучающем наборе (alpha=10): {:.2f}'.\
      format(ridge10.score(X_train, y_train)))
print('Правильность на тестовом наборе (alpha=10): {:.2f}'.\
      format(ridge10.score(X_test, y_test)))

print('Правильность на обучающем наборе (alpha=0.5): {:.2f}'.\
      format(ridge05.score(X_train, y_train)))
print('Правильность на тестовом наборе (alpha=0.5): {:.2f}'.\
      format(ridge05.score(X_test, y_test)))

print('Правильность на обучающем наборе (alpha=0.1): {:.2f}'.\
      format(ridge01.score(X_train, y_train)))
print('Правильность на тестовом наборе (alpha=0.1): {:.2f}'.\
      format(ridge01.score(X_test, y_test)))

print('Правильность на обучающем наборе (alpha=0.01): {:.2f}'.\
      format(ridge001.score(X_train, y_train)))
print('Правильность на тестовом наборе (alpha=0.01): {:.2f}'.\
      format(ridge001.score(X_test, y_test)))

plt.plot(ridge.coef_, 's', label='Гребневая регрессия alpha=1')
plt.plot(ridge10.coef_, '^', label='Гребневая регрессия alpha=10')
plt.plot(ridge01.coef_, 'v', label='Гребневая регрессия alpha=0.1')
plt.plot(lr.coef_, 'o', label='Линейная регрессия')

plt.xlabel('Индекс коэффициента')
plt.ylabel('Оценка коэффициента')
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
