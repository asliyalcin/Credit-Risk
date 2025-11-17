# utils/results.py

import os
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import joblib


@dataclass
class ModelResult:
    model_name: str
    best_threshold: float
    metrics: dict            # tn, fp, fn, tp, sensitivity, specificity, g_mean ...
    score: float             # main performance metric (e.g., g_mean)
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    model: object | None = None  # XGB, LGR, RF etc.


class ResultsRegistry:
    def __init__(self, run_id: str | None = None):
        # Unique ID for each run (versioning)
        if run_id is None:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.results: dict[str, ModelResult] = {}

    def add_model(self,
                  model_name: str,
                  best_threshold: float,
                  metrics: dict,
                  y_true,
                  y_pred,
                  y_proba,
                  model=None,
                  score: float | None = None):
        """
        Adds a model to the registry.

        If score is not provided, metrics["g_mean"] is used (if available).
        """
        if score is None and "g_mean" in metrics:
            score = metrics["g_mean"]

        result = ModelResult(
            model_name=model_name,
            best_threshold=float(best_threshold),
            metrics=metrics,
            score=float(score) if score is not None else np.nan,
            y_true=np.array(y_true),
            y_pred=np.array(y_pred),
            y_proba=np.array(y_proba),
            model=model
        )
        self.results[model_name] = result

    def save_all(self, base_dir: str = "results", save_models: bool = True):
        """
        Saves all models to disk:
        - results/<run_id>/<model_name>/metrics.json
        - results/<run_id>/<model_name>/predictions.csv
        - results/<run_id>/<model_name>/model.pkl
        """
        run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True)

        for name, res in self.results.items():
            model_dir = os.path.join(run_dir, name)
            os.makedirs(model_dir, exist_ok=True)

            # 1) metrics + threshold + score -> JSON
            metrics_clean = {}
            for k, v in res.metrics.items():
                if isinstance(v, (np.generic, np.number)):
                    v = v.item()
                metrics_clean[k] = v

            metrics_payload = {
                "model_name": res.model_name,
                "best_threshold": float(res.best_threshold),
                "score": float(res.score)
                if not isinstance(res.score, str) else res.score,
                "metrics": metrics_clean,
            }

            metrics_path = os.path.join(model_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=4, ensure_ascii=False)

            # 2) predictions csv
            df_preds = pd.DataFrame({
                "y_true": res.y_true,
                "y_pred": res.y_pred,
                "y_proba": res.y_proba,
            })
            preds_path = os.path.join(model_dir, "predictions.csv")
            df_preds.to_csv(preds_path, index=False)

            # 3) model.pkl
            if save_models and res.model is not None:
                model_path = os.path.join(model_dir, "model.pkl")
                joblib.dump(res.model, model_path)

        return run_dir
