"""
Detector de deriva de datos (Data Drift) para VinOps.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

class DriftDetector:
    def __init__(self, data_path: str = None, window_size: int = 20, eval_every: int = 10):
        self.data_path = data_path or os.getenv("DATA_PATH", "/app/data/wine_clean.csv")
        self.window_size = window_size
        self.eval_every = eval_every
        self.reference = self._load_reference()
        self.window = []
        self.prediction_count = 0
        self.last_eval = None

    def _load_reference(self):
        df = pd.read_csv(self.data_path)
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])
        cols = list(df.columns)
        return {
            "df": df,
            "columns": cols,
            "means": df.mean().to_dict(),
            "stds": df.std().to_dict(),
        }

    def add_sample(self, sample: Dict[str, float]):
        row = [float(sample.get(c, np.nan)) for c in self.reference["columns"]]
        self.window.append(row)
        self.prediction_count += 1
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def should_evaluate(self) -> bool:
        return (len(self.window) >= self.window_size and 
                self.prediction_count % self.eval_every == 0)

    def evaluate(self) -> Dict[str, Any]:
        if len(self.window) < self.window_size:
            return {"status": "INSUFFICIENT_DATA", "window_size": len(self.window)}

        window_df = pd.DataFrame(self.window, columns=self.reference["columns"])
        ref_df = self.reference["df"]

        # KS test
        ks_results = {}
        ks_alerts = []
        for col in self.reference["columns"]:
            try:
                stat, pvalue = stats.ks_2samp(ref_df[col].values, window_df[col].values)
                ks_results[col] = {
                    "statistic": round(float(stat), 4), 
                    "pvalue": round(float(pvalue), 4)
                }
                if pvalue < 0.05:
                    ks_alerts.append(col)
            except Exception:
                ks_results[col] = {"statistic": None, "pvalue": None}

        # PSI
        psi_results = {}
        psi_alerts = []
        for col in self.reference["columns"]:
            try:
                psi = self._compute_psi(ref_df[col].values, window_df[col].values)
                psi_results[col] = round(float(psi), 4)
                if psi > 0.25:
                    psi_alerts.append(col)
            except Exception:
                psi_results[col] = 0.0

        # Wasserstein
        wass = self._compute_wasserstein_multivariate(ref_df, window_df)

        num_ks = len(ks_alerts)
        num_psi = len(psi_alerts)

        if num_ks >= 3 or num_psi >= 2 or wass > 2.0:
            status = "DRIFT_DETECTED"
            recommendation = "RETRAIN"
        elif num_ks >= 1 or num_psi >= 1 or wass > 1.0:
            status = "MONITOR"
            recommendation = "MONITOR"
        else:
            status = "OK"
            recommendation = "OK"

        self.last_eval = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "window_size": len(self.window),
            "status": status,
            "recommendation": recommendation,
            "ks_test": {
                "alert_threshold": 0.05,
                "variables_alerted": ks_alerts,
                "num_alerted": num_ks,
                "details": ks_results,
            },
            "psi": {
                "alert_threshold": 0.25,
                "monitor_threshold": 0.1,
                "variables_alerted": psi_alerts,
                "num_alerted": num_psi,
                "details": psi_results,
            },
            "wasserstein": {
                "distance": round(float(wass), 4) if not (np.isnan(wass) or np.isinf(wass)) else 0.0,
                "alert_threshold": 2.0,
                "monitor_threshold": 1.0,
            }
        }
        return self.last_eval

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        if min_val == max_val or len(expected) == 0 or len(actual) == 0:
            return 0.0
        breakpoints = np.linspace(min_val, max_val, bins + 1)

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        expected_percents = np.clip(expected_percents, 0.0001, 1.0)
        actual_percents = np.clip(actual_percents, 0.0001, 1.0)

        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi)

    def _compute_wasserstein_multivariate(self, ref_df: pd.DataFrame, window_df: pd.DataFrame) -> float:
        distances = []
        for col in self.reference["columns"]:
            try:
                d = stats.wasserstein_distance(ref_df[col].values, window_df[col].values)
                std = self.reference["stds"][col]
                if std and std > 0 and not np.isnan(std):
                    d = d / std
                else:
                    d = 0.0
                if not (np.isnan(d) or np.isinf(d)):
                    distances.append(d)
            except Exception:
                distances.append(0.0)
        return float(np.mean(distances)) if distances else 0.0

    def get_summary(self) -> Dict[str, Any]:
        if self.last_eval is None:
            return {"status": "NOT_EVALUATED_YET", "window_size": len(self.window)}
        return self.last_eval
