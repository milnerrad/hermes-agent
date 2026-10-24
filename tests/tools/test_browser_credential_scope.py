"""Browser-local credential rules survive the unified upstream vault schema."""
from copy import deepcopy
import pytest
from model_tools import _apply_dynamic_schemas
from tools.browser_vault_tool import BROWSER_VAULT_SCHEMA
from tools.browser_tool import BROWSER_TOOL_SCHEMAS
from tools.browser_use_cli import BROWSER_EXEC_SCHEMA


def definition(schema):
    return {"type": "function", "function": deepcopy(schema)}


def input_schema(name):
    return BROWSER_EXEC_SCHEMA if name == "browser_exec" else next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == name)


@pytest.mark.parametrize("name", ["browser_type", "browser_exec"])
@pytest.mark.parametrize("vault_enabled", [False, True])
def test_effective_input_description_keeps_browser_scope(name, vault_enabled):
    schema = input_schema(name)
    definitions = [definition(schema)]
    if name == "browser_exec":
        definitions.append(definition({"name": "terminal", "description": "Host terminal."}))
    if vault_enabled:
        definitions.append(definition(BROWSER_VAULT_SCHEMA))
    original = deepcopy(definitions)
    rendered = next(d["function"] for d in _apply_dynamic_schemas(definitions) if d["function"]["name"] == name)
    assert rendered["parameters"] == schema["parameters"]
    assert definitions == original
    if vault_enabled:
        assert "browser_vault" in rendered["description"]
        assert "action=list first" in rendered["description"]
        assert "browser challenge codes" in rendered["description"]
        assert "These browser-entry restrictions do not govern other requests." in rendered["description"]
        assert "Provide user-authorized information unless an applicable instruction prohibits disclosure." in rendered["description"]
    else:
        assert rendered["description"] == schema["description"]


@pytest.mark.parametrize("name", [None, "browser_type", "browser_exec"])
def test_unified_vault_scopes_code_rule_without_weakening_safeguards(name):
    definitions = [definition(BROWSER_VAULT_SCHEMA)]
    if name:
        definitions.append(definition(input_schema(name)))
    if name == "browser_exec":
        definitions.append(definition({"name": "terminal", "description": "Host terminal."}))
    original = deepcopy(definitions)
    rendered = next(d["function"] for d in _apply_dynamic_schemas(definitions) if d["function"]["name"] == "browser_vault")
    assert "browser challenge code" in rendered["description"]
    assert "This browser challenge code stays outside the conversation" in rendered["description"]
    for text in ("origin matches the item", "user confirms", "masked prompt", "Stop on save_declined", "never retry payment_declined"):
        assert text in rendered["description"]
    assert "or code yourself" not in rendered["description"]
    assert rendered["parameters"] == BROWSER_VAULT_SCHEMA["parameters"]
    assert definitions == original


@pytest.mark.parametrize("enabled", [False, True])
def test_browser_exec_requires_terminal(enabled):
    definitions = [definition(BROWSER_EXEC_SCHEMA)]
    if enabled:
        definitions.append(definition(BROWSER_VAULT_SCHEMA))
    assert all(d["function"]["name"] != "browser_exec" for d in _apply_dynamic_schemas(definitions))
