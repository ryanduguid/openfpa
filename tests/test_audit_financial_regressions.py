"""Regression cases for the financial defects also found in the resumed fork."""
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from pyfpa import cashflow_from_config, load_config
from pyfpa.analysis.divestiture import Carveout, divest
from pyfpa.research.epochs import EpochEvaluation, ResearchEpoch, evaluate_challenger
from pyfpa.research.objective import MetricObjective, ResearchObjective
from pyfpa.research.registry import ModelRegistry, promote_challenger

CONFIG = Path(__file__).resolve().parents[1] / "examples/ridgeline/config.yaml"


@pytest.mark.parametrize("sale_month", [0, 6, 12])
def test_divestiture_preserves_kernel_pnl_identity(sale_month):
    forecast = cashflow_from_config(load_config(CONFIG))
    original = forecast.copy(deep=True)
    result = divest(
        forecast, Carveout(revenue=20000, gross_profit=8000, opex=2000),
        sale_month=sale_month, proceeds=0, annual_rate=0, tax_rate=0,
    )
    assert (result["revenue"] - result["cogs"] - result["gross_profit"]).abs().max() < 1e-8
    assert (result["cogs"].iloc[:sale_month] == forecast["cogs"].iloc[:sale_month]).all()
    assert (result["cogs"].iloc[sale_month:] == forecast["cogs"].iloc[sale_month:] - 12000).all()
    pd.testing.assert_frame_equal(forecast, original)


@pytest.mark.parametrize("field", [None, "channels", "opex", "debt", "working_capital", "opening_balances"])
def test_unknown_configuration_fields_are_rejected(tmp_path, field):
    values = load_config(CONFIG).model_dump()
    target = values if field is None else values[field]
    if isinstance(target, list):
        target = target[0]
    target["misspelt_field"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValidationError, match="misspelt_field"):
        load_config(path)


def test_correct_growth_applies_and_misspelt_growth_is_refused(tmp_path):
    values = load_config(CONFIG).model_dump()
    values["horizon_months"] = 24
    values["channels"] = [{
        "name": "Synthetic", "annual_revenue": 1200, "growth_rate": 0.1,
        "seasonality": [1] * 12, "cogs_pct": 0.5,
    }]
    path = tmp_path / "growth.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    revenue = cashflow_from_config(load_config(path))["revenue"]
    assert revenue.iloc[:12].sum() == pytest.approx(1200)
    assert revenue.iloc[12:].sum() == pytest.approx(1320)
    channel = values["channels"][0]
    channel["growth_rtae"] = channel.pop("growth_rate")
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValidationError, match="growth_rtae"):
        load_config(path)


@pytest.mark.parametrize("direction", ["lower", "higher"])
@pytest.mark.parametrize("challenger", [-1, 0, 1])
def test_zero_baseline_preserves_the_direction_of_change(direction, challenger):
    objective = ResearchObjective(metrics=[MetricObjective(name="metric", weight=1, direction=direction)])
    result = evaluate_challenger(objective, {"metric": 0}, {"metric": challenger}, [])
    expected = challenger if direction == "higher" else -challenger
    assert result.per_metric_improvement == {"metric": expected}
    assert result.promotion_eligible == (expected >= 0)


@pytest.mark.parametrize("champion,challenger", [
    (1, float("nan")), (float("nan"), 1), (float("inf"), 1),
    (1, float("-inf")), (-1e308, 1e308),
])
def test_non_finite_comparisons_are_refused_in_evaluation_and_recheck(champion, challenger):
    objective = ResearchObjective(metrics=[MetricObjective(name="metric", weight=1, direction="higher")])
    with pytest.raises(ValueError, match="finite"):
        evaluate_challenger(objective, {"metric": champion}, {"metric": challenger}, [])
    evaluation = EpochEvaluation(
        champion_metrics={"metric": champion}, challenger_metrics={"metric": challenger},
        per_metric_improvement={"metric": 1}, weighted_improvement=1, complexity_cost=0,
        objective_gain=1, hard_checks_passed=True, promotion_eligible=True,
    )
    epoch = ResearchEpoch(
        epoch_id="synthetic", created="2026-09-14", status="proposed", hypothesis="Synthetic check",
        champion_id="before", challenger_id="after", evaluation=evaluation,
    )
    with pytest.raises(ValueError, match="finite"):
        promote_challenger(
            ModelRegistry(), challenger_id="after", epoch=epoch,
            approved_by="Synthetic test marker", approved_at="2026-09-14", objective=objective,
        )
