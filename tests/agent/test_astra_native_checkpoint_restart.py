"""Exercise durable native checkpoint replay across an actual process boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-sol"])
def test_native_checkpoint_fresh_process_replay(tmp_path, model):
    db_path = tmp_path / "state.db"
    checkpoint = {
        "type": "compaction",
        "encrypted_content": "opaque-test-checkpoint-" * 1000,
        "_issuer_kind": "codex_backend",
    }
    with SessionDB(db_path=db_path) as db:
        db.create_session("restart", source="test")
        db.append_message("restart", "user", "Remember the assistant's fact")
        db.append_message("restart", "assistant", "fact present only before checkpoint")
        db.append_message("restart", "user", "Continue")
        db.append_message("restart", "assistant", "captured", codex_reasoning_items=[checkpoint])
    script = r'''
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from hermes_state import SessionDB
from run_agent import AIAgent

with SessionDB(db_path=Path(sys.argv[1])) as db:
    history = db.get_messages_as_conversation("restart")
checkpoint = history[-1]["codex_reasoning_items"][0]
with patch("model_tools.get_tool_definitions", return_value=[]):
    agent = AIAgent(
        model=sys.argv[2], provider="openai-codex", api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex", api_key="test-key",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[], save_trajectories=False,
    )
agent.codex_responses_native_compaction = True
agent.compression_enabled = True
agent.context_compressor.threshold_tokens = 130_000
agent.context_compressor.context_length = 200_000
agent.compression_idle_compact_after_seconds = 1
agent._last_activity_ts = 0
assert not agent.context_compressor.awaiting_real_usage_after_compression
response = SimpleNamespace(
    output=[SimpleNamespace(type="message", status="completed",
        content=[SimpleNamespace(type="output_text", text="Resumed")])],
    usage=SimpleNamespace(input_tokens=65_000, output_tokens=100, total_tokens=65_100),
    status="completed", incomplete_details=None, model=sys.argv[2],
)
with (
    patch("agent.codex_responses_adapter.estimate_native_responses_preflight_tokens", return_value=1_300_000),
    patch.object(agent, "_run_codex_stream", return_value=response) as provider,
    patch.object(agent, "_compress_context", side_effect=AssertionError("local compression before issuer usage")),
    patch.object(agent, "_persist_session"),
    patch.object(agent, "_save_trajectory"),
    patch.object(agent, "_cleanup_task_resources"),
):
    result = agent.run_conversation("continue", system_message="Stable original instruction", conversation_history=history)
assert result["completed"] and result["final_response"] == "Resumed"
provider.assert_called_once()
request = provider.call_args.args[0] if provider.call_args.args else provider.call_args.kwargs
wire = json.dumps(request)
assert checkpoint["encrypted_content"] in wire
assert "fact present only before checkpoint" not in wire
assert "Stable original instruction" in wire
assert "context_management" in request
assert not agent.context_compressor.awaiting_real_usage_after_compression
assert agent.context_compressor.last_prompt_tokens == 65_000
# The one-response latch must not permanently disable local fallback.
agent.context_compressor.note_request_rough_estimate(1_300_000)
agent.context_compressor.update_from_response({"prompt_tokens": agent.context_compressor.threshold_tokens + 100})
assert not agent.context_compressor.should_defer_preflight_to_real_usage(1_300_000)
agent.close()
print(json.dumps({"checkpoint": checkpoint, "provider_calls": provider.call_count}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(db_path), model],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(tmp_path / "child-home")},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt == {"checkpoint": checkpoint, "provider_calls": 1}
