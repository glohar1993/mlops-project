"""
Tier 2 — Fairness & Bias Monitor
===================================
Detects model bias across subgroups using:
  - Disparate Impact (4/5 rule): min_group_rate / max_group_rate < 0.8 → bias
  - Accuracy Gap: >10% accuracy difference across groups → bias
  - Demographic Parity: prediction rate disparity across protected attributes

References: Fairness-aware ML (Barocas et al.), NIST AI RMF
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime


class FairnessMonitor:
    """Compute fairness metrics across operation mode subgroups."""

    DISPARATE_IMPACT_THRESHOLD = 0.8    # 4/5 rule
    ACCURACY_GAP_THRESHOLD     = 0.10   # 10% max accuracy gap

    def __init__(self):
        self._history: List[Dict] = []

    def record(self, features: Dict[str, Any], prediction: int,
               ground_truth: Optional[int] = None) -> None:
        """Log a prediction (with optional ground truth) for bias tracking."""
        self._history.append({
            "features":     features,
            "prediction":   prediction,
            "ground_truth": ground_truth,
        })

    # ── Subgroup metrics ──────────────────────────────────────────

    def compute_subgroup_metrics(
        self,
        records: List[Dict],
        group_feature: str,
        positive_class: int = 0,   # 0 = "High" = favorable outcome
    ) -> Dict[str, Dict]:
        """Per-subgroup prediction rates and accuracy (if GT available)."""
        groups: Dict[str, List] = defaultdict(list)
        for r in records:
            key = str(r["features"].get(group_feature, "unknown"))
            groups[key].append(r)

        metrics: Dict[str, Dict] = {}
        for group, recs in groups.items():
            n     = len(recs)
            preds = [r["prediction"] for r in recs]
            pos_rate = sum(1 for p in preds if p == positive_class) / max(1, n)
            metrics[group] = {
                "count":        n,
                "positive_rate": round(pos_rate, 4),
                "predictions": {
                    "High":   sum(1 for p in preds if p == 0),
                    "Low":    sum(1 for p in preds if p == 1),
                    "Medium": sum(1 for p in preds if p == 2),
                },
            }
            truths  = [r["ground_truth"] for r in recs if r["ground_truth"] is not None]
            preds_gt = [r["prediction"]  for r in recs if r["ground_truth"] is not None]
            if truths:
                correct = sum(p == t for p, t in zip(preds_gt, truths))
                metrics[group]["accuracy"] = round(correct / len(truths), 4)

        return metrics

    # ── Fairness tests ────────────────────────────────────────────

    def compute_disparate_impact(self, metrics: Dict[str, Dict]) -> Dict:
        """Disparate Impact = min_positive_rate / max_positive_rate (4/5 rule)."""
        if len(metrics) < 2:
            return {"disparate_impact": 1.0, "status": "OK",
                    "reason": "insufficient_groups"}

        rates   = [v["positive_rate"] for v in metrics.values()]
        min_r, max_r = min(rates), max(rates)

        if max_r == 0:
            return {"disparate_impact": 1.0, "status": "OK",
                    "reason": "no_positive_predictions"}

        di     = min_r / max_r
        status = "BIAS_DETECTED" if di < self.DISPARATE_IMPACT_THRESHOLD else "OK"
        return {
            "disparate_impact": round(di, 4),
            "status":           status,
            "min_group_rate":   min_r,
            "max_group_rate":   max_r,
            "threshold":        self.DISPARATE_IMPACT_THRESHOLD,
        }

    def compute_accuracy_gap(self, metrics: Dict[str, Dict]) -> Dict:
        """Accuracy gap across subgroups."""
        accs = {k: v["accuracy"] for k, v in metrics.items()
                if "accuracy" in v}
        if len(accs) < 2:
            return {"accuracy_gap": 0.0, "status": "OK",
                    "reason": "insufficient_ground_truth_data"}

        vals = list(accs.values())
        gap  = max(vals) - min(vals)
        status = "BIAS_DETECTED" if gap > self.ACCURACY_GAP_THRESHOLD else "OK"
        return {
            "accuracy_gap": round(gap, 4),
            "best_group":   max(accs, key=accs.get),
            "worst_group":  min(accs, key=accs.get),
            "status":       status,
            "threshold":    self.ACCURACY_GAP_THRESHOLD,
        }

    # ── Full report ───────────────────────────────────────────────

    def full_report(
        self,
        records:       Optional[List[Dict]] = None,
        group_feature: str = "Operation_Mode",
    ) -> Dict:
        """Generate a full fairness audit report."""
        data = records if records is not None else self._history
        if not data:
            return {"overall_status": "NO_DATA", "records_analyzed": 0}

        subgroup = self.compute_subgroup_metrics(data, group_feature)
        di       = self.compute_disparate_impact(subgroup)
        ag       = self.compute_accuracy_gap(subgroup)

        overall = "OK"
        if di["status"] == "BIAS_DETECTED" or ag["status"] == "BIAS_DETECTED":
            overall = "BIAS_DETECTED"

        return {
            "overall_status":    overall,
            "records_analyzed":  len(data),
            "group_feature":     group_feature,
            "subgroup_metrics":  subgroup,
            "disparate_impact":  di,
            "accuracy_gap":      ag,
            "generated_at":      datetime.utcnow().isoformat(),
        }
