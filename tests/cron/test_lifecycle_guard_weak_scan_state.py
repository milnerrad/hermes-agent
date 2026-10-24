"""A lenient interpreter-source visit must not suppress an executed-script scan (SEC-1)."""

import shlex

import pytest

from cron import lifecycle_guard


def _mention_and_execute(target, executed_first):
    mention = f"python3 - <<'PY'\n{str(target)!r}\nPY\n"
    execute = f"sh {shlex.quote(str(target))}\n"
    return execute + mention if executed_first else mention + execute


@pytest.mark.parametrize("executed_first", [False, True], ids=["mention-first", "execute-first"])
@pytest.mark.parametrize("limit", ["file-size", "text-budget"])
def test_weak_visit_cannot_hide_executed_scan_refusal(tmp_path, executed_first, limit):
    target = tmp_path / "script.sh"
    padding = "# harmless padding\n" * 60000 if limit == "file-size" else "#" + "x" * 70000 + "\n"
    target.write_text(padding + "hermes gateway restart\n", encoding="utf-8")

    # Each visit gets a fresh top-level budget. A source-only mention remains
    # lenient; executing the identical file must enforce the established cap.
    mention = f"python3 - <<'PY'\n{str(target)!r}\nPY\n"
    assert lifecycle_guard.scan_gateway_lifecycle(mention, cwd=str(tmp_path)) == (False, None)
    unsafe, refusal = lifecycle_guard.scan_gateway_lifecycle(
        _mention_and_execute(target, executed_first), cwd=str(tmp_path)
    )
    assert unsafe is True, "a weaker source visit must not hide an executed unscannable script"
    assert refusal is not None
    assert ("scan cap" if limit == "file-size" else "budget") in refusal


@pytest.mark.parametrize("executed_first", [False, True], ids=["mention-first", "execute-first"])
@pytest.mark.parametrize("lifecycle", [False, True], ids=["ordinary", "lifecycle"])
def test_small_shared_reference_retains_supported_verdict(tmp_path, executed_first, lifecycle):
    target = tmp_path / "script.sh"
    target.write_text("hermes gateway restart\n" if lifecycle else "printf ordinary\n", encoding="utf-8")

    assert lifecycle_guard.scan_gateway_lifecycle(
        _mention_and_execute(target, executed_first), cwd=str(tmp_path)
    ) == (lifecycle, None)
