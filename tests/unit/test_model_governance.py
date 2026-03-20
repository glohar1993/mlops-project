"""
Unit Tests — Model Governance (src/model_governance.py)
=========================================================
Tests: ModelCard CRUD, DataLineageTracker, ApprovalWorkflow
All permutations: create/save/load, approve/reject, lineage hashing
"""

import os
import json
import pytest
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Isolate governance dir in temp
TEMP_GOV_DIR = tempfile.mkdtemp()
os.environ["GOVERNANCE_DIR"] = TEMP_GOV_DIR

from src.model_governance import ModelCard, DataLineageTracker, ApprovalWorkflow

# Reset class-level paths
DataLineageTracker.LINEAGE_FILE = os.path.join(TEMP_GOV_DIR, "data_lineage.json")
ApprovalWorkflow.APPROVAL_FILE  = os.path.join(TEMP_GOV_DIR, "approvals.json")


@pytest.fixture(autouse=True)
def cleanup():
    """Clean governance files between tests."""
    yield
    for f in ["data_lineage.json", "approvals.json"]:
        p = os.path.join(TEMP_GOV_DIR, f)
        if os.path.exists(p):
            os.remove(p)
    for f in os.listdir(TEMP_GOV_DIR):
        if f.startswith("model_card_"):
            os.remove(os.path.join(TEMP_GOV_DIR, f))


# ════════════════════════════════════════════════════════════════
#  ModelCard
# ════════════════════════════════════════════════════════════════

class TestModelCard:

    def test_create_card(self):
        card = ModelCard(model_version=1, run_id="run-abc")
        assert card.model_version == 1
        assert card.card["run_id"] == "run-abc"

    def test_set_performance(self):
        card = ModelCard(1)
        card.set_performance({"accuracy": 0.92, "f1_score": 0.91})
        assert card.card["performance_metrics"]["accuracy"] == 0.92

    def test_set_data_lineage(self):
        card = ModelCard(1)
        card.set_data_lineage("abc123", "/data/train.csv", 10000)
        assert card.card["data_lineage"]["data_hash"] == "abc123"
        assert card.card["data_lineage"]["row_count"]  == 10000

    def test_set_features(self):
        card = ModelCard(1)
        feats = ["Temperature_C", "Vibration_Hz", "Operation_Mode"]
        card.set_features(feats)
        assert card.card["input_features"] == feats

    def test_add_limitation(self):
        card = ModelCard(1)
        card.add_limitation("Not validated on Asian markets")
        card.add_limitation("Requires re-training if Operation_Mode extends")
        assert len(card.card["limitations"]) == 2

    def test_set_fairness(self):
        card = ModelCard(1)
        card.set_fairness({"overall_status": "OK", "disparate_impact": 0.95})
        assert card.card["fairness_assessment"]["overall_status"] == "OK"

    def test_save_creates_file(self):
        card = ModelCard(42)
        path = card.save()
        assert os.path.exists(path)

    def test_save_and_load_roundtrip(self):
        card = ModelCard(7, run_id="run-007")
        card.set_performance({"accuracy": 0.88})
        card.add_limitation("Simulated data only")
        card.save()

        loaded = ModelCard.load(7)
        assert loaded is not None
        assert loaded.card["run_id"] == "run-007"
        assert loaded.card["performance_metrics"]["accuracy"] == 0.88
        assert "Simulated data only" in loaded.card["limitations"]

    def test_load_nonexistent_returns_none(self):
        assert ModelCard.load(9999) is None

    def test_to_dict_returns_copy(self):
        card = ModelCard(1)
        d    = card.to_dict()
        d["model_version"] = 999
        assert card.card["model_version"] == 1  # original unchanged

    @pytest.mark.parametrize("version", [1, 5, 10, 100])
    def test_multiple_model_versions(self, version):
        card = ModelCard(version)
        card.save()
        loaded = ModelCard.load(version)
        assert loaded.model_version == version

    def test_schema_version_present(self):
        card = ModelCard(1)
        assert "schema_version" in card.card

    def test_approval_status_defaults_pending(self):
        card = ModelCard(1)
        assert card.card["approval_status"] == "pending"


# ════════════════════════════════════════════════════════════════
#  Data Lineage Tracker
# ════════════════════════════════════════════════════════════════

class TestDataLineageTracker:

    def test_record_creates_entry(self):
        entry = DataLineageTracker.record(
            model_version=1,
            data_path="artifacts/raw/data.csv",
            metrics={"accuracy": 0.87},
        )
        assert entry["model_version"] == 1
        assert entry["metrics"]["accuracy"] == 0.87

    def test_get_returns_recorded_entry(self):
        DataLineageTracker.record(2, "data.csv", {"accuracy": 0.90})
        entry = DataLineageTracker.get(2)
        assert entry is not None
        assert entry["model_version"] == 2

    def test_get_nonexistent_returns_none(self):
        assert DataLineageTracker.get(9999) is None

    def test_multiple_versions_stored_separately(self):
        DataLineageTracker.record(10, "data_v1.csv", {"accuracy": 0.80})
        DataLineageTracker.record(11, "data_v2.csv", {"accuracy": 0.85})
        all_entries = DataLineageTracker.get_all()
        assert "10" in all_entries
        assert "11" in all_entries
        assert all_entries["10"]["metrics"]["accuracy"] == 0.80
        assert all_entries["11"]["metrics"]["accuracy"] == 0.85

    def test_file_not_found_returns_sentinel(self):
        result = DataLineageTracker.compute_data_hash("/no/such/file.csv")
        assert result == "file-not-found"

    def test_hash_real_file(self):
        path = "artifacts/raw/data.csv"
        if os.path.exists(path):
            h = DataLineageTracker.compute_data_hash(path)
            assert len(h) == 32   # MD5 hex digest
            # Deterministic
            assert DataLineageTracker.compute_data_hash(path) == h

    def test_lineage_persisted_to_disk(self):
        DataLineageTracker.record(99, "data.csv", {"accuracy": 0.91})
        assert os.path.exists(DataLineageTracker.LINEAGE_FILE)
        with open(DataLineageTracker.LINEAGE_FILE) as f:
            data = json.load(f)
        assert "99" in data


# ════════════════════════════════════════════════════════════════
#  Approval Workflow
# ════════════════════════════════════════════════════════════════

class TestApprovalWorkflow:

    def test_request_creates_pending(self):
        req = ApprovalWorkflow.request_approval(1, "ml-engineer", "New v1 ready")
        assert req["status"] == "pending"
        assert req["requested_by"] == "ml-engineer"

    def test_approve_changes_status(self):
        ApprovalWorkflow.request_approval(2, "engineer", "v2 ready")
        result = ApprovalWorkflow.approve(2, "lead-reviewer")
        assert result["status"]      == "approved"
        assert result["approved_by"] == "lead-reviewer"
        assert result["approved_at"] is not None

    def test_reject_changes_status(self):
        ApprovalWorkflow.request_approval(3, "engineer", "v3 ready")
        result = ApprovalWorkflow.reject(3, "lead", "accuracy < threshold")
        assert result["status"]          == "rejected"
        assert result["rejected_reason"] == "accuracy < threshold"

    def test_is_approved_true_after_approval(self):
        ApprovalWorkflow.request_approval(4, "eng", "v4")
        ApprovalWorkflow.approve(4, "lead")
        assert ApprovalWorkflow.is_approved(4) is True

    def test_is_approved_false_pending(self):
        ApprovalWorkflow.request_approval(5, "eng", "v5")
        assert ApprovalWorkflow.is_approved(5) is False

    def test_is_approved_false_rejected(self):
        ApprovalWorkflow.request_approval(6, "eng", "v6")
        ApprovalWorkflow.reject(6, "lead", "reason")
        assert ApprovalWorkflow.is_approved(6) is False

    def test_approve_nonexistent_raises(self):
        with pytest.raises(ValueError, match="No approval request"):
            ApprovalWorkflow.approve(9999, "lead")

    def test_reject_nonexistent_raises(self):
        with pytest.raises(ValueError, match="No approval request"):
            ApprovalWorkflow.reject(9999, "lead", "reason")

    def test_get_status_returns_dict(self):
        ApprovalWorkflow.request_approval(7, "eng", "reason")
        status = ApprovalWorkflow.get_status(7)
        assert isinstance(status, dict)
        assert status["model_version"] == 7

    def test_get_status_nonexistent_returns_none(self):
        assert ApprovalWorkflow.get_status(9998) is None

    @pytest.mark.parametrize("version,requestor,reason", [
        (10, "eng-a", "Scheduled retrain v10"),
        (11, "eng-b", "Drift-triggered v11"),
        (12, "eng-c", "Hyperparameter tuned v12"),
    ])
    def test_multiple_concurrent_approvals(self, version, requestor, reason):
        ApprovalWorkflow.request_approval(version, requestor, reason)
        ApprovalWorkflow.approve(version, "approver")
        assert ApprovalWorkflow.is_approved(version)
