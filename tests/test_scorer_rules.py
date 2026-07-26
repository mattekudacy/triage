from __future__ import annotations

from triage.scorer.rules import RulesRiskScorer
from triage.taxonomy import Step
from triage.trajectory import Trajectory


def make_step(action="test step", tool_called=None, error=None, idempotent=False):
    return Step(index=0, action=action, tool_called=tool_called, error=error, idempotent=idempotent)


def traj(*steps):
    t = Trajectory()
    for s in steps:
        t.append(s)
    return t


def test_low_risk_read_only_step():
    scorer = RulesRiskScorer()
    step = make_step(action="search results", tool_called="search", idempotent=True)
    result = scorer(step, traj(step))
    assert result.score < 0.5


def test_high_risk_email_action():
    scorer = RulesRiskScorer()
    step = make_step(action="send_email to user@example.com")
    result = scorer(step, traj(step))
    assert result.score >= 0.9
    assert result.reason is not None


def test_high_risk_payment_tool():
    scorer = RulesRiskScorer()
    step = make_step(tool_called="charge_card")
    result = scorer(step, traj(step))
    assert result.score >= 0.9


def test_high_risk_delete_action():
    scorer = RulesRiskScorer()
    step = make_step(action="DELETE FROM users WHERE id=1")
    result = scorer(step, traj(step))
    assert result.score >= 0.9


def test_high_risk_drop_table():
    scorer = RulesRiskScorer()
    step = make_step(action="DROP TABLE payments")
    result = scorer(step, traj(step))
    assert result.score >= 0.9


def test_medium_risk_external_write():
    scorer = RulesRiskScorer()
    step = make_step(action="write file to disk", tool_called="file_write")
    result = scorer(step, traj(step))
    assert result.score >= 0.6


def test_non_idempotent_flag_raises_base_score():
    scorer = RulesRiskScorer()
    step = make_step(action="update_record", idempotent=False)
    result = scorer(step, traj(step))
    assert result.score == 0.2


def test_idempotent_flag_keeps_score_low():
    scorer = RulesRiskScorer()
    step = make_step(action="get_data", idempotent=True)
    result = scorer(step, traj(step))
    assert result.score == 0.0


def test_custom_high_risk_patterns():
    scorer = RulesRiskScorer(high_risk_patterns=["nuke_db", "wipe_all"])
    step = make_step(action="nuke_db")
    result = scorer(step, traj(step))
    assert result.score >= 0.9


def test_custom_medium_risk_patterns():
    scorer = RulesRiskScorer(medium_risk_patterns=["sync_data", "refresh_cache"])
    step = make_step(action="sync_data")
    result = scorer(step, traj(step))
    assert result.score >= 0.6


def test_score_returns_risk_score_instance():
    from triage.scorer.base import RiskScore

    scorer = RulesRiskScorer()
    step = make_step()
    result = scorer(step, traj(step))
    assert isinstance(result, RiskScore)
