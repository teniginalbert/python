import mglearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso().fit(X_train, y_train)

print('Правильность на обучающем наборе: {:.2f}'.\
      format(lasso.score(X_train, y_train)))
print('Правильность на контрольном наборе: {:.2f}'.\
      format(lasso.score(X_test, y_test)))
print('Количество использованных признаков: {}'.\
      format(np.sum(lasso.coef_ != 0)))

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)

print('Правильность на обучающем наборе (alpha=0.01): {:.2f}'.\
      format(lasso001.score(X_train, y_train)))
print('Правильность на контрольном наборе(alpha=0.01): {:.2f}'.\
      format(lasso001.score(X_test, y_test)))
print('Количество использованных признаков(alpha=0.01): {}'.\
      format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)

print('Правильность на обучающем наборе (alpha=0.0001): {:.2f}'.\
      format(lasso00001.score(X_train, y_train)))
print('Правильность на контрольном наборе(alpha=0.0001): {:.2f}'.\
      format(lasso00001.score(X_test, y_test)))
print('Количество использованных признаков(alpha=0.0001): {}'.\
      format(np.sum(lasso00001.coef_ != 0)))


plt.plot(lasso.coef_, 's', label='Лассо alpha=1')
plt.plot(lasso001.coef_, '^', label='Лассо alpha=0.01')
plt.plot(lasso00001.coef_, 'v', label='Лассо alpha=0.0001')

plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel('Индекс коэффициента')
plt.ylabel('Оценка коэффициента')