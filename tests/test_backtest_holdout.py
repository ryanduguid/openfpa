from unittest.mock import Mock

import pytest
from pyfpa.config.schemas import EntityConfig
from pyfpa.backtest.holdout import holdout_backtest


def _actuals():
    out = {}
    cash = 0.0
    for i in range(6):
        cash += 20.0
        out[f"2026-{i+1:02d}"] = {"revenue": 100.0, "gross_profit": 40.0,
                                  "ebitda": 30.0, "ending_cash": cash}
    return out


def _build_cfg(growth, *, horizon_months=3):
    def _fn(fit_actuals):
        n = len(fit_actuals)
        last_rev = list(fit_actuals.values())[-1]["revenue"]
        annual = last_rev * 12 * (1 + growth)
        return EntityConfig.model_validate({
            "name": "h", "start_month": f"2026-{n+1:02d}", "horizon_months": horizon_months,
            "tax_rate": 0.0,
            "channels": [{"name": "C", "annual_revenue": annual, "growth_rate": 0.0,
                          "seasonality": [1.0] * 12, "cogs_pct": 0.6}],
            "opex": [], "debt": [],
            "working_capital": {"dso_days": 0, "dpo_days": 0, "dio_days": 0},
            "opening_balances": {"cash": 80.0},
        })
    return _fn


def test_holdout_backtest_scores_holdout():
    res = holdout_backtest(_actuals(), _build_cfg(0.0), holdout=3,
                           score_lines=["revenue", "ebitda"])
    assert "revenue" in res.per_line
    assert res.fitness >= 0.0


def test_holdout_discriminates_better_assumption():
    good = holdout_backtest(_actuals(), _build_cfg(0.0), holdout=3, score_lines=["revenue"])
    bad = holdout_backtest(_actuals(), _build_cfg(0.5), holdout=3, score_lines=["revenue"])
    assert good.fitness < bad.fitness


@pytest.mark.parametrize("period_count,holdout", [(6, 6), (6, 7), (1, 1), (1, 3), (0, 1), (0, 0)])
def test_holdout_rejects_too_few_periods(period_count, holdout):
    actuals = dict(list(_actuals().items())[:period_count])
    build = Mock(side_effect=AssertionError("insufficient periods reached the builder"))
    with pytest.raises(ValueError) as exc:
        holdout_backtest(actuals, build, holdout=holdout, score_lines=["revenue"])
    assert str(exc.value) == f"need more than {holdout} periods, got {period_count}"
    build.assert_not_called()


@pytest.mark.parametrize("holdout", [0, -1, -6, -7])
def test_holdout_rejects_non_positive_before_build(holdout):
    build = Mock(side_effect=AssertionError("invalid holdout reached the builder"))
    with pytest.raises(ValueError, match="^holdout must be at least 1$"):
        holdout_backtest(_actuals(), build, holdout=holdout, score_lines=["revenue"])
    build.assert_not_called()


@pytest.mark.parametrize("holdout", [1, 3, 5])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_holdout_valid_boundaries_preserve_order(holdout, reverse_order):
    rows = list(_actuals().items())
    if reverse_order:
        rows.reverse()
    actuals = dict(rows)
    fit_count = len(actuals) - holdout
    expected_fit = dict(rows[:fit_count])
    build = Mock(side_effect=_build_cfg(0.0, horizon_months=holdout))

    result = holdout_backtest(actuals, build, holdout=holdout, score_lines=["revenue"])

    build.assert_called_once_with(expected_fit)
    assert list(build.call_args.args[0]) == list(expected_fit)
    assert result.fitness == pytest.approx(0.0)
    assert result.per_line == pytest.approx({"revenue": 0.0})
    assert result.weights == {"revenue": 1.0}
