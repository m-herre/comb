"""Stream rows in, then forget some: each edit equals building the final rows in one go."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from comb import CombClassifier

X, y = make_classification(12000, 15, n_informative=8, random_state=0)
X_test, y_test = X[10000:], y[10000:]
batches = [(X[i:i + 2000], y[i:i + 2000]) for i in range(0, 10000, 2000)]

clf = CombClassifier().fit(*batches[0])
for Xb, yb in batches[1:]:
    clf.add(Xb, yb)
    print(f"{clf.n_rows_:>6} rows  AUC {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.4f}")

# a deletion request for one batch
clf.forget(*batches[2])
reference = CombClassifier().fit(*batches[0])
for i in (1, 3, 4):
    reference.add(*batches[i])
diff = np.abs(clf.predict_proba(X_test) - reference.predict_proba(X_test)).max()
print(f"after forget: {clf.n_rows_} rows, max |difference| to never adding it: {diff:.1e}")
