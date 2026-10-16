"""Build shards separately (e.g. at different sites) and merge them: only cell statistics move, never rows."""
import numpy as np
from sklearn.datasets import make_classification

from comb import CombClassifier

X, y = make_classification(9000, 12, n_informative=6, n_classes=3, random_state=1)
X_test = X[8000:]
seed, site_a, site_b = (X[:1000], y[:1000]), (X[1000:4500], y[1000:4500]), (X[4500:8000], y[4500:8000])

model = CombClassifier().fit(*seed)          # a shared, representative sample fixes the frozen state
shard_a = model.empty_like().add(*site_a)    # at site A
shard_b = model.empty_like().add(*site_b)    # at site B
model.merge(shard_a).merge(shard_b)

one_go = CombClassifier().fit(*seed).add(np.vstack([site_a[0], site_b[0]]), np.concatenate([site_a[1], site_b[1]]))
diff = np.abs(model.predict_proba(X_test) - one_go.predict_proba(X_test)).max()
print(f"{model.n_rows_} rows; merged vs one go: max |difference| {diff:.1e}")
