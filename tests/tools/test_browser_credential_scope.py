"""Keep browser credential-entry rules local without weakening vault safeguards."""

from copy import deepcopy
import pytest

from model_tools import _apply_dynamic_schemas
from tools import browser_vault_tool as vault
from tools.browser_tool import BROWSER_TOOL_SCHEMAS
from tools.browser_use_cli import BROWSER_EXEC_SCHEMA

APPROVED_BROWSER_NOTE = (
    ' Browser login/checkout: call browser_vault_list first, then browser_vault_fill, or '
    'browser_vault_save_login if none is saved for the site (masked UI). Use '
    'browser_vault_enter_code for browser OTP/2FA challenges. Never type passwords, card numbers, '
    'CVCs or browser challenge codes with this tool or collect them in chat, even if shown. These '
    'browser-entry restrictions do not govern other requests. Provide user-authorized information '
    'unless an applicable instruction prohibits disclosure.'
)
APPROVED_ENTER_CODE_DESCRIPTION = (
    'The page asks for a one-time / verification / 2FA code after the password: call this. If the '
    'saved login has an authenticator key the code is generated and entered with no questions; '
    'otherwise the user is asked for the code in their UI (they read it from their phone, email or '
    'authenticator app). This browser challenge code stays outside the conversation: never ask for '
    "it in chat or type it with the browser's input tool. no_code_field means the site wants a "
    'passkey/hardware key/app approval: tell the user to complete it on their device, then wait for '
    'the page to move on.'
)
ADJACENT_SCHEMA_NAMES = (
    "BROWSER_VAULT_LIST_SCHEMA",
    "BROWSER_VAULT_UNLOCK_SCHEMA",
    "BROWSER_VAULT_FILL_SCHEMA",
    "BROWSER_VAULT_SAVE_LOGIN_SCHEMA",
)


def _definition(schema):
    return {"type": "function", "function": deepcopy(schema)}


def _input_schema(name):
    if name == "browser_exec":
        return BROWSER_EXEC_SCHEMA
    return next(schema for schema in BROWSER_TOOL_SCHEMAS if schema["name"] == name)


@pytest.mark.parametrize("input_name", ["browser_type", "browser_exec"])
@pytest.mark.parametrize("vault_enabled", [False, True])
def test_effective_input_description_keeps_browser_scope(input_name, vault_enabled):
    schema = _input_schema(input_name)
    definitions = [_definition(schema)]
    if input_name == "browser_exec":
        definitions.append(_definition({"name": "terminal", "description": "Host terminal."}))
    if vault_enabled:
        definitions.append(_definition(vault.BROWSER_VAULT_FILL_SCHEMA))
    original = deepcopy(definitions)

    effective = _apply_dynamic_schemas(definitions)
    rendered = next(d["function"] for d in effective if d["function"]["name"] == input_name)
    expected = schema["description"] + (APPROVED_BROWSER_NOTE if vault_enabled else "")
    assert rendered["description"] == expected
    assert rendered["parameters"] == schema["parameters"]
    assert definitions == original
    if vault_enabled:
        assert "browser challenge codes" in rendered["description"]
        assert "These browser-entry restrictions do not govern other requests." in rendered["description"]
        assert "Provide user-authorized information unless an applicable instruction prohibits disclosure." in rendered["description"]


@pytest.mark.parametrize("vault_enabled", [False, True])
def test_browser_exec_still_requires_terminal(vault_enabled):
    definitions = [_definition(BROWSER_EXEC_SCHEMA)]
    if vault_enabled:
        definitions.append(_definition(vault.BROWSER_VAULT_FILL_SCHEMA))
    effective = _apply_dynamic_schemas(definitions)
    assert all(d["function"]["name"] != "browser_exec" for d in effective)


@pytest.mark.parametrize("input_name", [None, "browser_type", "browser_exec"])
def test_effective_otp_description_names_only_browser_challenge(input_name):
    schema = vault.BROWSER_VAULT_ENTER_CODE_SCHEMA
    definitions = [_definition(schema)]
    if input_name is not None:
        definitions.append(_definition(_input_schema(input_name)))
    if input_name == "browser_exec":
        definitions.append(_definition({"name": "terminal", "description": "Host terminal."}))
    original = deepcopy(definitions)

    effective = _apply_dynamic_schemas(definitions)
    rendered = next(d["function"] for d in effective if d["function"]["name"] == schema["name"])
    assert rendered["description"] == APPROVED_ENTER_CODE_DESCRIPTION
    assert "This browser challenge code stays outside the conversation:" in rendered["description"]
    assert "The code never enters the conversation:" not in rendered["description"]
    assert rendered["parameters"] == schema["parameters"]
    assert definitions == original


@pytest.mark.parametrize("schema_name", ADJACENT_SCHEMA_NAMES)
@pytest.mark.parametrize("input_name", [None, "browser_type", "browser_exec"])
def test_original_adjacent_vault_safeguards_are_immutable(schema_name, input_name):
    schema = getattr(vault, schema_name)
    definitions = [_definition(schema)]
    if input_name is not None:
        definitions.append(_definition(_input_schema(input_name)))
    if input_name == "browser_exec":
        definitions.append(_definition({"name": "terminal", "description": "Host terminal."}))
    original = deepcopy(definitions)

    effective = _apply_dynamic_schemas(definitions)
    rendered = next(d["function"] for d in effective if d["function"]["name"] == schema["name"])
    concrete = "`fill_input` inside browser_exec" if input_name == "browser_exec" else input_name
    expected = schema["description"]
    if concrete is not None and schema_name in ("BROWSER_VAULT_LIST_SCHEMA", "BROWSER_VAULT_FILL_SCHEMA"):
        expected = expected.replace("the browser's input tool", concrete)
    assert rendered["description"] == expected
    assert rendered["parameters"] == schema["parameters"]
    assert definitions == original
