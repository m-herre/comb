"""Fit, predict, score."""
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from comb import CombClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = CombClassifier().fit(X_train, y_train)
print(f"AUC {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.4f}")
