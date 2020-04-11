from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = \
train_test_split(cancer.data, cancer.target, \
                 stratify=cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print('Правильность на обучающем наборе: {:.3f}'.\
format(gbrt.score(X_train, y_train)))
print('Правильность на тестовом наборе: {:.3f}'.\
format(gbrt.score(X_test, y_test)))

gbrt_depth = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt_depth.fit(X_train, y_train)

print('Правильность на обучающем наборе (max_depth=1): {:.3f}'.\
format(gbrt_depth.score(X_train, y_train)))
print('Правильность на тестовом наборе (max_depth=1): {:.3f}'.\
format(gbrt_depth.score(X_test, y_test)))

gbrt_rate = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt_rate.fit(X_train, y_train)

print('Правильность на обучающем наборе (learning_rate=0.01): {:.3f}'.\
format(gbrt_rate.score(X_train, y_train)))
print('Правильность на тестовом наборе (learning_rate=0.01): {:.3f}'.\
format(gbrt_rate.score(X_test, y_test)))
