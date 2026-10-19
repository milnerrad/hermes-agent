"""Pending #95295 direct callers must not erase executable interpreter source."""
import pytest
from cron.lifecycle_guard import contains_gateway_lifecycle_command, contains_gateway_lifecycle_command_or_referenced_script
from tools.approval_detection import _is_shell_token_spliced_gateway_lifecycle


@pytest.mark.parametrize("scan", [contains_gateway_lifecycle_command, contains_gateway_lifecycle_command_or_referenced_script, _is_shell_token_spliced_gateway_lifecycle])
@pytest.mark.parametrize("command", [
    "python3 - <<'PY'\nimport os; os.system(\"hermes gateway restart\")\nPY",
    "osascript <<'APPLESCRIPT'\ndo shell script \"hermes gateway restart\"\nAPPLESCRIPT",
])
def test_interpreter_heredoc_reaches_direct_and_full_scanners(scan, command):
    assert scan(command)


@pytest.mark.parametrize("scan", [contains_gateway_lifecycle_command, contains_gateway_lifecycle_command_or_referenced_script, _is_shell_token_spliced_gateway_lifecycle])
def test_plain_quoted_cat_runbook_remains_data(scan):
    assert not scan("cat > runbook.txt <<'EOF'\nhermes gateway restart\nEOF")


def test_default_shell_masking_contract_stays_separate():
    from tools.shell_heredoc import strip_inert_heredoc_bodies
    command = "python3 - <<'PY'\nprint('hermes gateway restart')\nPY"
    assert "hermes gateway restart" not in strip_inert_heredoc_bodies(command)
