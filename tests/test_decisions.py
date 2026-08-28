from __future__ import annotations

import pytest


def test_cost_sensitive_threshold_minimizes_expected_loss_and_reports_budget_metrics() -> None:
    from fraudshield.decision import DecisionCosts, optimize_decisions

    result = optimize_decisions(
        probabilities=[0.9, 0.8, 0.4, 0.2],
        labels=[1, 0, 1, 0],
        costs=DecisionCosts(false_negative=100.0, false_positive=10.0),
        analyst_alert_budget=2,
    )

    assert result.chosen_threshold == 0.4
    assert result.expected_loss == 10.0
    assert result.precision == 2 / 3
    assert result.recall == 1.0
    assert result.f1 == 0.8
    assert result.alert_count == 3
    assert result.budget_alert_count == 2
    assert result.budget_precision == 0.5
    assert result.budget_recall == 0.5
    assert [point.threshold for point in result.threshold_sweep] == [1.0, 0.9, 0.8, 0.4, 0.2, 0.0]


def test_all_equal_scores_use_costs_and_deterministic_tie_breaks() -> None:
    from fraudshield.decision import DecisionCosts, optimize_decisions

    alert_all = optimize_decisions(
        probabilities=[0.5, 0.5, 0.5],
        labels=[1, 0, 0],
        costs=DecisionCosts(false_negative=100.0, false_positive=30.0),
        analyst_alert_budget=1,
    )
    suppress_alerts = optimize_decisions(
        probabilities=[0.5, 0.5, 0.5],
        labels=[1, 0, 0],
        costs=DecisionCosts(false_negative=10.0, false_positive=30.0),
        analyst_alert_budget=1,
    )

    assert alert_all.chosen_threshold == 0.5
    assert alert_all.expected_loss == 60.0
    assert alert_all.recall == 1.0
    assert alert_all.precision == 1 / 3
    assert alert_all.budget_precision == 1.0
    assert alert_all.budget_recall == 1.0
    assert suppress_alerts.chosen_threshold == 1.0
    assert suppress_alerts.alert_count == 0
    assert suppress_alerts.expected_loss == 10.0


@pytest.mark.parametrize(
    ("probabilities", "labels", "costs", "budget", "message"),
    [
        ([], [], (100.0, 10.0), 1, "probabilities must not be empty"),
        ([0.7], [1, 0], (100.0, 10.0), 1, "same length"),
        ([1.1], [1], (100.0, 10.0), 1, "between 0 and 1"),
        ([0.7], [2], (100.0, 10.0), 1, "only 0 and 1"),
        ([0.7], [1], (0.0, 10.0), 1, "false_negative"),
        ([0.7], [1], (100.0, -1.0), 1, "false_positive"),
        ([0.7], [1], (100.0, 10.0), 0, "analyst_alert_budget"),
        ([0.7], [1], (100.0, 10.0), 2, "analyst_alert_budget"),
    ],
)
def test_decision_inputs_reject_invalid_costs_arrays_and_budgets(
    probabilities: list[float],
    labels: list[int],
    costs: tuple[float, float],
    budget: int,
    message: str,
) -> None:
    from fraudshield.decision import DecisionCosts, optimize_decisions

    with pytest.raises(ValueError, match=message):
        optimize_decisions(
            probabilities=probabilities,
            labels=labels,
            costs=DecisionCosts(false_negative=costs[0], false_positive=costs[1]),
            analyst_alert_budget=budget,
        )
