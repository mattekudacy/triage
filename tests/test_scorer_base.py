from triage.scorer.base import RiskScore, StepRiskScorer
from triage.taxonomy import Step
from triage.trajectory import Trajectory


def test_risk_score_fields():
    rs = RiskScore(score=0.8, reason="destructive action")
    assert rs.score == 0.8
    assert rs.reason == "destructive action"


def test_risk_score_defaults():
    rs = RiskScore(score=0.1)
    assert rs.reason is None


def test_risk_score_clamps_above_one():
    rs = RiskScore(score=1.5)
    assert rs.score == 1.0


def test_risk_score_clamps_below_zero():
    rs = RiskScore(score=-0.1)
    assert rs.score == 0.0


def test_step_risk_scorer_is_callable_protocol():

    def my_scorer(step: Step, trajectory: Trajectory) -> RiskScore:
        return RiskScore(score=0.0)

    assert isinstance(my_scorer, StepRiskScorer)
