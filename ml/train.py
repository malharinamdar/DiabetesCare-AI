"""Train the DiabetesCare risk model and export it for the web app.

    pip install -r ml/requirements.txt
    python ml/train.py

Same recipe as v1 (notebook in ml/notebooks/): dedupe, recategorise smoking
history, SMOTE (0.1) + random under-sampling (0.5), random forest with
min_samples_leaf=4 / min_samples_split=5. Changes: trees are depth-capped so
the forest can be served from a serverless function, and the scaler is gone
(trees are scale-invariant).

Writes:
  lib/model/forest.json   trees with per-node P(diabetes) for path attributions
  lib/model/cohort.json   per-feature histograms (diabetic vs not) for the charts
  ml/metrics.json         held-out scores shown on the site
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lib" / "model"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension",
            "heart_disease", "gender_male", "smoking_current", "smoking_past"]
SMOKING = {"never": "never", "No Info": "never", "current": "current",
           "ever": "past", "former": "past", "not current": "past"}

df = pd.read_csv(ROOT / "ml" / "data" / "diabetes_prediction_dataset.csv")
df = df[df.gender != "Other"].drop_duplicates()
df["smoking"] = df.smoking_history.map(SMOKING)
df["gender_male"] = (df.gender == "Male").astype(float)
df["smoking_current"] = (df.smoking == "current").astype(float)
df["smoking_past"] = (df.smoking == "past").astype(float)

X, y = df[FEATURES].astype(float), df.diabetes
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def make(depth):
    return Pipeline([
        ("over", SMOTE(sampling_strategy=0.1, random_state=42)),
        ("under", RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=4,
                                      min_samples_split=5, random_state=42, n_jobs=-1)),
    ])


# The resampled training set is 1/3 positive, so raw forest scores overstate risk.
# A prior shift maps them back to the real prevalence for the displayed probability.
train_prior = float(ytr.mean())
resampled_prior = 1 / 3


def to_population(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * (train_prior / (1 - train_prior)) / (resampled_prior / (1 - resampled_prior))
    return odds / (1 + odds)


def evaluate(model):
    s = model.predict_proba(Xte)[:, 1]
    pred = (s >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(yte, pred), 4),
        "precision": round(precision_score(yte, pred), 4),
        "recall": round(recall_score(yte, pred), 4),
        "f1": round(f1_score(yte, pred), 4),
        "roc_auc": round(roc_auc_score(yte, s), 4),
        "brier_population": round(brier_score_loss(yte, to_population(s)), 4),
        "nodes": int(sum(e.tree_.node_count for e in model.named_steps["rf"].estimators_)),
    }


uncapped = evaluate(make(None).fit(Xtr, ytr))
model = make(10).fit(Xtr, ytr)
capped = evaluate(model)
print("uncapped", uncapped)
print("depth 10", capped)


def export_tree(est):
    t = est.tree_
    frac = t.value[:, 0, :] / t.value[:, 0, :].sum(axis=1, keepdims=True)
    return {
        "f": t.feature.tolist(),
        "t": t.threshold.tolist(),
        "l": t.children_left.tolist(),
        "r": t.children_right.tolist(),
        "v": [round(float(x), 5) for x in frac[:, 1]],
    }


trees = [export_tree(e) for e in model.named_steps["rf"].estimators_]


def score_exported(x):
    x = np.asarray(x, dtype=np.float32).astype(np.float64)  # sklearn compares float32 inputs in float64
    total = 0.0
    for tr in trees:
        i = 0
        while tr["l"][i] != -1:
            i = tr["l"][i] if x[tr["f"][i]] <= tr["t"][i] else tr["r"][i]
        total += tr["v"][i]
    return total / len(trees)


sample = Xte.values[:500]
ref = model.predict_proba(sample)[:, 1]
mine = np.array([score_exported(r) for r in sample])
d = np.abs(ref - mine)
assert d.max() < 1e-3, "exported forest disagrees with sklearn"

(OUT / "forest.json").write_text(json.dumps({
    "features": FEATURES,
    "trees": trees,
    "trainPrior": round(train_prior, 5),
    "resampledPrior": resampled_prior,
}, separators=(",", ":")))

# Cohort histograms for the "where you sit" charts (all de-duplicated rows).
BINS = {
    "age": np.arange(0, 85, 5).tolist(),
    "bmi": np.arange(10, 62, 2.5).tolist(),
    "HbA1c_level": np.arange(3.5, 9.25, 0.25).tolist(),
    "blood_glucose_level": np.arange(80, 310, 10).tolist(),
}
cohort = {"n": int(len(df)), "positives": int(df.diabetes.sum()), "features": {}}
for f, edges in BINS.items():
    pos = np.histogram(df.loc[df.diabetes == 1, f].clip(edges[0], edges[-1] - 1e-9), bins=edges)[0]
    neg = np.histogram(df.loc[df.diabetes == 0, f].clip(edges[0], edges[-1] - 1e-9), bins=edges)[0]
    cohort["features"][f] = {
        "edges": [round(e, 2) for e in edges],
        "pos": (pos / pos.sum()).round(5).tolist(),
        "neg": (neg / neg.sum()).round(5).tolist(),
    }
(OUT / "cohort.json").write_text(json.dumps(cohort, separators=(",", ":")))

metrics = {
    "rows_after_dedupe": int(len(df)),
    "prevalence": round(float(y.mean()), 4),
    "test_size": int(len(Xte)),
    "served_model": {"max_depth": 10, **capped},
    "uncapped_reference": {"max_depth": None, **uncapped},
    "v1_reported_accuracy": 0.94,
}
(ROOT / "ml" / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT / "metrics.json").write_text(json.dumps(metrics))
print(json.dumps(metrics, indent=2))
print("forest.json", round((OUT / "forest.json").stat().st_size / 1024), "KB")
