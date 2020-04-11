
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = \
train_test_split(cancer.data, cancer.target, \
                 stratify=cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print('Правильность на обучающем наборе: {:.3f}'.\
format(forest.score(X_train, y_train)))
print('Правильность на тестовом наборе: {:.3f}'.\
format(forest.score(X_test, y_test)))
