import pytest
import yaml
from pydantic import ValidationError

from pyfpa import (
    cash13_forecast, cashflow_from_config, load_cash13_config, load_config,
    runway_summary,
)
from pyfpa.cash13.schemas import Cash13Config, WeeklyFlow
from pyfpa.config.schemas import (
    Channel, DebtInstrument, EntityConfig, OpeningBalances, OpexLine,
    WorkingCapitalConfig,
)


NON_FINITE = [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "-Infinity"]


@pytest.fixture
def monthly_data():
    return {
        "name": "Synthetic Co",
        "start_month": "2026-01",
        "horizon_months": 12,
        "tax_rate": 0,
        "channels": [{
            "name": "Sales", "annual_revenue": 1200,
            "seasonality": [1] * 12, "cogs_pct": 0,
        }],
        "opex": [{"name": "Rent", "kind": "fixed", "monthly_amount": 20}],
        "debt": [{
            "name": "Loan", "kind": "term_loan",
            "opening_balance": 0, "annual_rate": 0,
        }],
        "working_capital": {"dso_days": 0, "dpo_days": 0, "dio_days": 0},
        "opening_balances": {},
    }


@pytest.fixture
def weekly_data():
    return {
        "opening_cash": 100, "weeks": 2,
        "receipts": [{"name": "Sales", "amount": 20, "start_week": 1}],
        "disbursements": [{"name": "Costs", "amount": 0, "start_week": 1}],
    }


MONTHLY_FIELDS = [
    (model, path, field)
    for model, path, fields in [
        (EntityConfig, (), ["tax_rate", "da_monthly", "capex_monthly"]),
        (Channel, ("channels", 0), [
            "annual_revenue", "growth_rate", "seasonality", "cogs_pct",
        ]),
        (OpexLine, ("opex", 0), ["monthly_amount", "pct_of_revenue"]),
        (DebtInstrument, ("debt", 0), [
            "opening_balance", "annual_rate", "monthly_principal",
        ]),
        (WorkingCapitalConfig, ("working_capital",), [
            "dso_days", "dpo_days", "dio_days",
        ]),
        (OpeningBalances, ("opening_balances",), ["cash", "ar", "ap", "inventory", "nol"]),
    ]
    for field in fields
]


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize("model,path,field", MONTHLY_FIELDS)
def test_monthly_schemas_reject_non_finite_inputs(monthly_data, model, path, field, value):
    data = monthly_data
    for key in path:
        data = data[key]
    data[field] = [value] + [1] * 11 if field == "seasonality" else value
    location = (field, 0) if field == "seasonality" else (field,)
    with pytest.raises(ValidationError) as exc:
        model(**data)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(location, "finite_number")]


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize("model,field,data", [
    (Cash13Config, "opening_cash", {"opening_cash": 100}),
    (WeeklyFlow, "amount", {"name": "Sales", "amount": 20, "start_week": 1}),
])
def test_weekly_schemas_reject_non_finite_inputs(model, field, data, value):
    with pytest.raises(ValidationError) as exc:
        model(**(data | {field: value}))
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [((field,), "finite_number")]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("path", [
    ("channels", 0, "annual_revenue"),
    ("channels", 0, "growth_rate"),
    ("channels", 0, "seasonality", 5),
    ("opex", 0, "monthly_amount"),
    ("opex", 0, "pct_of_revenue"),
    ("debt", 0, "opening_balance"),
    ("working_capital", "dso_days"),
    ("opening_balances", "cash"),
    ("capex_monthly",),
])
def test_monthly_yaml_rejects_non_finite_inputs(tmp_path, monthly_data, path, value):
    target = monthly_data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path = tmp_path / "monthly.yaml"
    config_path.write_text(yaml.safe_dump(monthly_data), encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        load_config(config_path)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(path, "finite_number")]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("path", [
    ("opening_cash",), ("receipts", 0, "amount"), ("disbursements", 0, "amount"),
])
def test_weekly_yaml_rejects_non_finite_inputs(tmp_path, weekly_data, path, value):
    target = weekly_data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text(yaml.safe_dump(weekly_data), encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        load_cash13_config(config_path)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(path, "finite_number")]


def test_finite_monthly_inputs_preserve_cashflow(tmp_path, monthly_data):
    config_path = tmp_path / "monthly.yaml"
    config_path.write_text(yaml.safe_dump(monthly_data), encoding="utf-8")
    forecast = cashflow_from_config(load_config(config_path))
    assert forecast["revenue"].tolist() == [100] * 12
    assert forecast["opex"].tolist() == [20] * 12
    assert forecast["ending_cash"].tolist() == [80 * month for month in range(1, 13)]


def test_finite_negative_inputs_and_seasonal_zero_remain_valid(monthly_data):
    monthly_data["channels"][0].update(growth_rate=-0.5, seasonality=[0] + [1] * 11)
    monthly_data["opex"][0]["monthly_amount"] = -20
    monthly_data["opex"].append({"name": "Credit", "kind": "variable", "pct_of_revenue": -0.1})
    monthly_data["opening_balances"].update(cash=-100, ar=-10, ap=-20, inventory=-5)
    cfg = EntityConfig.model_validate(monthly_data)
    assert cfg.channels[0].seasonality[0] == 0
    assert cfg.channels[0].growth_rate == -0.5
    assert cfg.opex[0].monthly_amount == -20
    assert cfg.opex[1].pct_of_revenue == -0.1
    assert cfg.opening_balances.model_dump() == {
        "cash": -100, "ar": -10, "ap": -20, "inventory": -5, "nol": 0,
    }


def test_finite_weekly_inputs_preserve_trajectory_and_extra_keys(tmp_path, weekly_data):
    weekly_data["opening_cash"] = -10
    weekly_data["note"] = "ignored"
    weekly_data["receipts"][0]["note"] = "ignored"
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text(yaml.safe_dump(weekly_data), encoding="utf-8")
    cfg = load_cash13_config(config_path)
    forecast = cash13_forecast(cfg)
    assert forecast["ending_cash"].tolist() == [10, 10]
    assert runway_summary(forecast) == {"min_cash": 10, "min_week": 1, "first_negative_week": None}
    assert "note" not in cfg.model_dump()
    assert "note" not in cfg.receipts[0].model_dump()
