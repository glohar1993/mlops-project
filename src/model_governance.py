"""
Tier 1 — Model Governance
===========================
- ModelCard        : document model purpose, metrics, limitations, fairness
- DataLineageTracker: record which data version trained which model (MD5 hash)
- ApprovalWorkflow : human sign-off before Production promotion

Storage: JSON files in artifacts/governance/  (extend to PostgreSQL in prod)
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List

GOVERNANCE_DIR = os.getenv("GOVERNANCE_DIR", "artifacts/governance")
os.makedirs(GOVERNANCE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  Model Card
# ══════════════════════════════════════════════════════════════════

class ModelCard:
    """Documents a model's characteristics, limitations, and intended use."""

    def __init__(self, model_version: int, run_id: str = "unknown"):
        self.model_version = model_version
        self.card: Dict[str, Any] = {
            "schema_version":   "1.0",
            "model_version":    model_version,
            "run_id":           run_id,
            "created_at":       datetime.utcnow().isoformat(),
            "model_type":       "LogisticRegression",
            "task":             "multiclass_classification",
            "classes":          ["High", "Low", "Medium"],
            "intended_use":     "Predict industrial machine efficiency status for predictive maintenance",
            "out_of_scope_use": "Not for safety-critical decisions without human review",
            "input_features":   [],
            "performance_metrics": {},
            "data_lineage":     {},
            "limitations":      [],
            "fairness_assessment": {},
            "approval_status":  "pending",
            "approved_by":      None,
            "approved_at":      None,
        }

    def set_performance(self, metrics: Dict[str, float]) -> "ModelCard":
        self.card["performance_metrics"] = metrics
        return self

    def set_data_lineage(self, data_hash: str, data_path: str,
                         row_count: int) -> "ModelCard":
        self.card["data_lineage"] = {
            "data_hash":   data_hash,
            "data_path":   data_path,
            "row_count":   row_count,
            "recorded_at": datetime.utcnow().isoformat(),
        }
        return self

    def set_features(self, features: List[str]) -> "ModelCard":
        self.card["input_features"] = features
        return self

    def add_limitation(self, limitation: str) -> "ModelCard":
        self.card["limitations"].append(limitation)
        return self

    def set_fairness(self, assessment: Dict) -> "ModelCard":
        self.card["fairness_assessment"] = assessment
        return self

    def save(self) -> str:
        path = os.path.join(GOVERNANCE_DIR, f"model_card_v{self.model_version}.json")
        with open(path, "w") as f:
            json.dump(self.card, f, indent=2)
        return path

    @classmethod
    def load(cls, model_version: int) -> Optional["ModelCard"]:
        path = os.path.join(GOVERNANCE_DIR, f"model_card_v{model_version}.json")
        if not os.path.exists(path):
            return None
        obj = cls(model_version)
        with open(path) as f:
            obj.card = json.load(f)
        return obj

    def to_dict(self) -> Dict:
        return self.card.copy()


# ══════════════════════════════════════════════════════════════════
#  Data Lineage Tracker
# ══════════════════════════════════════════════════════════════════

class DataLineageTracker:
    """Track data version → model version relationships."""

    LINEAGE_FILE = os.path.join(GOVERNANCE_DIR, "data_lineage.json")

    @staticmethod
    def compute_data_hash(data_path: str) -> str:
        """MD5 content hash of data file for reproducibility."""
        if not os.path.exists(data_path):
            return "file-not-found"
        h = hashlib.md5()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def record(cls, model_version: int, data_path: str,
               metrics: Dict, environment: str = "unknown") -> Dict:
        """Record a data → model lineage entry."""
        entry = {
            "model_version": model_version,
            "data_path":     data_path,
            "data_hash":     cls.compute_data_hash(data_path),
            "trained_at":    datetime.utcnow().isoformat(),
            "environment":   environment,
            "metrics":       metrics,
        }
        lineage = cls._load()
        lineage[str(model_version)] = entry
        with open(cls.LINEAGE_FILE, "w") as f:
            json.dump(lineage, f, indent=2)
        return entry

    @classmethod
    def get(cls, model_version: int) -> Optional[Dict]:
        return cls._load().get(str(model_version))

    @classmethod
    def get_all(cls) -> Dict:
        return cls._load()

    @classmethod
    def _load(cls) -> Dict:
        if os.path.exists(cls.LINEAGE_FILE):
            with open(cls.LINEAGE_FILE) as f:
                return json.load(f)
        return {}


# ══════════════════════════════════════════════════════════════════
#  Approval Workflow
# ══════════════════════════════════════════════════════════════════

class ApprovalWorkflow:
    """Require human sign-off before promoting a model to Production."""

    APPROVAL_FILE = os.path.join(GOVERNANCE_DIR, "approvals.json")

    @classmethod
    def request_approval(cls, model_version: int, requested_by: str,
                         reason: str) -> Dict:
        """Create a pending approval request."""
        req = {
            "model_version":   model_version,
            "requested_by":    requested_by,
            "reason":          reason,
            "status":          "pending",
            "requested_at":    datetime.utcnow().isoformat(),
            "approved_by":     None,
            "approved_at":     None,
            "rejected_reason": None,
        }
        data = cls._load()
        data[str(model_version)] = req
        cls._save(data)
        return req

    @classmethod
    def approve(cls, model_version: int, approved_by: str) -> Dict:
        data = cls._load()
        key  = str(model_version)
        if key not in data:
            raise ValueError(f"No approval request for model v{model_version}")
        data[key]["status"]      = "approved"
        data[key]["approved_by"] = approved_by
        data[key]["approved_at"] = datetime.utcnow().isoformat()
        cls._save(data)
        return data[key]

    @classmethod
    def reject(cls, model_version: int, rejected_by: str, reason: str) -> Dict:
        data = cls._load()
        key  = str(model_version)
        if key not in data:
            raise ValueError(f"No approval request for model v{model_version}")
        data[key]["status"]          = "rejected"
        data[key]["approved_by"]     = rejected_by
        data[key]["approved_at"]     = datetime.utcnow().isoformat()
        data[key]["rejected_reason"] = reason
        cls._save(data)
        return data[key]

    @classmethod
    def is_approved(cls, model_version: int) -> bool:
        return cls._load().get(str(model_version), {}).get("status") == "approved"

    @classmethod
    def get_status(cls, model_version: int) -> Optional[Dict]:
        return cls._load().get(str(model_version))

    @classmethod
    def _load(cls) -> Dict:
        if os.path.exists(cls.APPROVAL_FILE):
            with open(cls.APPROVAL_FILE) as f:
                return json.load(f)
        return {}

    @classmethod
    def _save(cls, data: Dict) -> None:
        with open(cls.APPROVAL_FILE, "w") as f:
            json.dump(data, f, indent=2)
