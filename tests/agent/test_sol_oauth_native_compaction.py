"""Exact Sol native compaction eligibility on official Codex OAuth only."""
from types import SimpleNamespace
import pytest
from agent.native_compaction import native_compaction_context_management, resolve_native_compaction_capabilities

CODEX = "https://chatgpt.com/backend-api/codex"

@pytest.mark.parametrize("model,provider,url,eligible", [
    ("gpt-6.1-sol", "openai-codex", CODEX, True),
    ("gpt-6.1-sol-900k", "openai-codex", CODEX, True),
    ("GPT-6.1-SOL", "openai-codex", CODEX + "/", True),
    ("gpt-6.1-sol-mini", "openai-codex", CODEX, False),
    ("gpt-6.1-sol-preview", "openai-codex", CODEX, False),
    ("gpt-6.1-sol", "openai", "https://api.openai.com/v1", False),
    ("gpt-6.1-sol", "openai", CODEX, False),
    ("gpt-6.1-sol", "openai-codex", "https://relay.example/v1", False),
    ("gpt-6.1-sol", "openai-codex", "https://chatgpt.com.example/backend-api/codex", False),
    ("gpt-6.1-sol", "openai-codex", "http://chatgpt.com/backend-api/codex", False),
    ("gpt-6.1-sol", "openai-codex", None, False),
])
def test_sol_capability_and_request_gate(model, provider, url, eligible):
    codex = provider == "openai-codex"
    caps = resolve_native_compaction_capabilities(model=model, provider=provider, base_url=url, is_codex_backend=codex)
    assert caps["native_compaction"] is eligible
    agent = SimpleNamespace(model=model, provider=provider, base_url=url,
        codex_responses_native_compaction=True, compression_enabled=True,
        capabilities={"openai_native_compaction": True})
    for runtime in (None, caps):
        agent.runtime_capabilities = runtime
        assert (native_compaction_context_management(agent, is_codex_backend=codex) is not None) is eligible

@pytest.mark.parametrize("field", ["codex_responses_native_compaction", "compression_enabled", "compression_checkpoint_required"])
def test_sol_optout_and_checkpoint_gate(field):
    agent = SimpleNamespace(model="gpt-6.1-sol", provider="openai-codex", base_url=CODEX,
        codex_responses_native_compaction=True, compression_enabled=True,
        compression_checkpoint_required=False)
    setattr(agent, field, field == "compression_checkpoint_required")
    assert native_compaction_context_management(agent, is_codex_backend=True) is None
