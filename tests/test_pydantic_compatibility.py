import json

import pytest
import yaml
from pydantic import BaseModel, create_model

from pyfpa.memory.onboarding import ArchitectureProposal
from pyfpa.research.registry import ModelVersion, load_model_registry, save_model_registry


@pytest.mark.parametrize("model,field,data,required,field_type", [
    (ArchitectureProposal, "model_components", {
        "summary": "Synthetic model", "model_components": ["Revenue", "Cash"],
    }, False, "array"),
    (ModelVersion, "model_id", {
        "model_id": "model-v1", "created": "2026-01-01", "artifact": "model.py",
    }, True, "string"),
])
def test_model_fields_keep_public_names(model, field, data, required, field_type):
    assert not hasattr(BaseModel, field)
    assert field in model.model_fields
    cfg = model(**data)
    assert getattr(cfg, field) == data[field]
    assert cfg == model.model_validate(data)
    assert cfg.model_dump()[field] == data[field]
    assert cfg.model_dump(by_alias=True)[field] == data[field]
    assert json.loads(cfg.model_dump_json())[field] == data[field]
    assert model.model_validate(cfg.model_dump()) == cfg
    for mode in ("validation", "serialization"):
        schema = model.model_json_schema(mode=mode)
        assert schema["properties"][field]["type"] == field_type
        assert (field in schema.get("required", [])) is required


@pytest.mark.parametrize("model", [ArchitectureProposal, ModelVersion])
@pytest.mark.parametrize("field", ["model_validate", "model_dump"])
def test_validation_and_dump_members_remain_protected(model, field):
    with pytest.raises((NameError, ValueError), match="protected namespace"):
        create_model("MemberCollision", __base__=model, **{field: (str, ...)})


@pytest.mark.parametrize("model", [ArchitectureProposal, ModelVersion])
@pytest.mark.parametrize("field", ["model_validate_custom", "model_dump_custom"])
def test_validation_and_dump_prefixes_remain_protected(model, field):
    with pytest.warns(UserWarning, match="protected namespace"):
        create_model("NamespaceCollision", __base__=model, **{field: (str, ...)})


def test_registry_preserves_model_id_in_existing_yaml(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("""schema_version: 1
champion:
  model_id: champion-v1
  created: '2026-01-01'
  artifact: champion.py
challengers:
  - model_id: challenger-v2
    created: '2026-02-01'
    artifact: challenger.py
retired:
  - model_id: retired-v0
    created: '2025-12-01'
    artifact: retired.py
promotions: []
""", encoding="utf-8")
    registry = load_model_registry(path)
    assert registry.champion.model_id == "champion-v1"
    assert registry.challengers[0].model_id == "challenger-v2"
    assert registry.retired[0].model_id == "retired-v0"
    save_model_registry(registry, path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["champion"]["model_id"] == "champion-v1"
    assert saved["challengers"][0]["model_id"] == "challenger-v2"
    assert saved["retired"][0]["model_id"] == "retired-v0"
    assert load_model_registry(path) == registry
