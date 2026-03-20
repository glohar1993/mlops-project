"""
Governance Tests — Model Lifecycle Governance
===============================================
Tests: complete governance lifecycle, lineage → approval → deployment,
       multiple model versions, audit trail completeness
"""

import sys
import os
import pytest
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

TEMP_GOV = tempfile.mkdtemp()
os.environ["GOVERNANCE_DIR"] = TEMP_GOV

from src.model_governance import ModelCard, DataLineageTracker, ApprovalWorkflow
DataLineageTracker.LINEAGE_FILE = os.path.join(TEMP_GOV, "data_lineage.json")
ApprovalWorkflow.APPROVAL_FILE  = os.path.join(TEMP_GOV, "approvals.json")


@pytest.fixture(autouse=True)
def clean():
    yield
    import glob
    for f in glob.glob(os.path.join(TEMP_GOV, "*.json")):
        os.remove(f)


class TestFullGovernanceLifecycle:
    """Tests the end-to-end governance workflow for a new model version."""

    def test_full_lifecycle_v1(self):
        """
        Lifecycle:
        1. Record data lineage (data hash + row count)
        2. Create model card (performance metrics, features, limitations)
        3. Request approval
        4. Approve
        5. Verify model is approved
        """
        version = 1
        metrics = {"accuracy": 0.88, "f1_score": 0.87}

        # Step 1: Record lineage
        lineage = DataLineageTracker.record(version, "artifacts/raw/data.csv", metrics)
        assert lineage["model_version"] == version

        # Step 2: Create model card
        card = ModelCard(version, run_id="run-001")
        card.set_performance(metrics)
        card.set_features(["Temperature_C", "Operation_Mode"])
        card.add_limitation("Trained on simulated data — validate on real production")
        card.set_fairness({"overall_status": "OK"})
        path = card.save()
        assert os.path.exists(path)

        # Step 3: Request approval
        req = ApprovalWorkflow.request_approval(version, "ml-engineer", "v1 metrics pass threshold")
        assert req["status"] == "pending"

        # Step 4: Approve
        ApprovalWorkflow.approve(version, "lead-ml-engineer")
        assert ApprovalWorkflow.is_approved(version)

        # Step 5: Verify card survives reload
        loaded = ModelCard.load(version)
        assert loaded.card["performance_metrics"]["accuracy"] == 0.88

    def test_rejected_model_not_approved(self):
        version = 2
        ApprovalWorkflow.request_approval(version, "eng", "v2 ready")
        ApprovalWorkflow.reject(version, "lead", "accuracy 0.73 < threshold 0.75")
        assert ApprovalWorkflow.is_approved(version) is False

    def test_multiple_versions_governance(self):
        """Three model versions with independent governance states."""
        versions_data = [
            (10, 0.88, "approved", "lead"),
            (11, 0.91, "approved", "lead"),
            (12, 0.74, "rejected", "lead"),
        ]
        for v, acc, action, reviewer in versions_data:
            DataLineageTracker.record(v, "data.csv", {"accuracy": acc})
            ApprovalWorkflow.request_approval(v, "eng", f"v{v}")
            if action == "approved":
                ApprovalWorkflow.approve(v, reviewer)
            else:
                ApprovalWorkflow.reject(v, reviewer, f"accuracy {acc} < 0.75")

        assert ApprovalWorkflow.is_approved(10) is True
        assert ApprovalWorkflow.is_approved(11) is True
        assert ApprovalWorkflow.is_approved(12) is False

    def test_lineage_data_hash_recorded(self):
        """Data hash must be recorded so we can reproduce exact model."""
        DataLineageTracker.record(20, "artifacts/raw/data.csv", {"accuracy": 0.85})
        entry = DataLineageTracker.get(20)
        assert "data_hash" in entry
        assert entry["data_hash"] != ""

    def test_model_card_limitations_required(self):
        """Production models must document at least one limitation."""
        card = ModelCard(30)
        card.add_limitation("Not validated on extreme operating conditions")
        assert len(card.card["limitations"]) >= 1

    def test_approval_audit_trail_complete(self):
        """Approval record must have requestor, approver, timestamps."""
        ApprovalWorkflow.request_approval(40, "ml-eng", "v40 ready")
        ApprovalWorkflow.approve(40, "principal-eng")
        status = ApprovalWorkflow.get_status(40)
        assert status["requested_by"] == "ml-eng"
        assert status["approved_by"]  == "principal-eng"
        assert status["requested_at"] is not None
        assert status["approved_at"]  is not None

    def test_governance_without_approval_flag(self):
        """Model without approval request is not approved."""
        assert ApprovalWorkflow.is_approved(9999) is False

    @pytest.mark.parametrize("accuracy,should_pass", [
        (0.80, True),   # above 0.75 threshold
        (0.75, True),   # exactly at threshold
        (0.74, False),  # below threshold
    ])
    def test_accuracy_threshold_governance(self, accuracy, should_pass):
        from src.feature_registry import MIN_ACCURACY
        above_threshold = accuracy >= MIN_ACCURACY
        assert above_threshold == should_pass
