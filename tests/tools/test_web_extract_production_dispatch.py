"""Stable extraction contracts through the actual public tool, not the retired dispatcher."""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tools import web_result_cache as cache
from tools import web_tools as wt
from tools import web_tools_extract as wte

URL = 'https://example.com/start'


@pytest.fixture
def dispatch(monkeypatch):
    config = {'extract_timeout': 0.02, 'keyless_fallback': False, 'keyless_rescue': False}
    monkeypatch.setattr(wt, '_load_web_config', lambda: config)
    monkeypatch.setattr(wt, '_get_extract_backend', lambda: 'primary')
    monkeypatch.setattr(wt, '_ensure_web_plugins_loaded', lambda: None)
    monkeypatch.setattr(wt, 'async_is_safe_url', AsyncMock(return_value=True))
    monkeypatch.setattr(wt, '_get_secondary_provider', lambda *args: None)
    monkeypatch.setattr(wte, '_rescue_eligible', lambda p: False)
    monkeypatch.setattr('tools.website_policy.check_website_access', lambda u: None)
    real_get, real_put = cache.extract_cache_get, cache.extract_cache_put
    monkeypatch.setattr(cache, 'extract_cache_get', lambda *a, **kw: None)
    put = Mock()
    monkeypatch.setattr(cache, 'extract_cache_put', put)
    monkeypatch.setattr(wt._debug, 'log_call', lambda *a, **kw: None)
    monkeypatch.setattr(wt._debug, 'save', lambda: None)

    async def run(provider, urls=None):
        monkeypatch.setattr(wt, '_resolve_extract_provider', lambda b: (provider, None))
        response = await asyncio.wait_for(wt.web_extract_tool(urls or [URL]), timeout=0.5)
        return json.loads(response)

    return SimpleNamespace(run=run, config=config, put=put, real_get=real_get, real_put=real_put)


@pytest.mark.asyncio
@pytest.mark.parametrize('sync', [False, True], ids=['async', 'sync-thread'])
async def test_public_tool_returns_per_url_timeout_without_rescue(dispatch, sync):
    release = threading.Event()

    async def hang(urls, format=None):
        await asyncio.sleep(10)

    def sync_hang(urls, format=None):
        release.wait(2)

    provider = SimpleNamespace(name='primary', extract=sync_hang if sync else hang)
    try:
        result = await dispatch.run(provider)
    finally:
        release.set()
    row = result['results'][0]
    assert row['url'] == URL
    assert 'timed out' in row['error'].lower()
    assert 'primary' in row['error']
    dispatch.put.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('sync', [False, True], ids=['async', 'sync-thread'])
async def test_primary_timeout_reaches_configured_secondary(dispatch, monkeypatch, sync):
    release = threading.Event()

    async def hang(urls, format=None):
        await asyncio.sleep(10)

    def sync_hang(urls, format=None):
        release.wait(2)

    secondary = SimpleNamespace(name='secondary', extract=AsyncMock(return_value=[
        {'url': URL, 'content': 'secondary content'},
    ]))
    monkeypatch.setattr(wt, '_get_secondary_provider', lambda *args: secondary)
    try:
        result = await dispatch.run(SimpleNamespace(name='primary', extract=sync_hang if sync else hang))
    finally:
        release.set()
    row = result['results'][0]
    assert row['content'] == 'secondary content'
    metadata = secondary.extract.return_value[0]['metadata']
    assert metadata['served_by'] == 'secondary'
    assert 'timed out' in metadata['backend_error']
    secondary.extract.assert_awaited_once_with([URL], format=None)
    dispatch.put.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('sync', [False, True], ids=['async', 'sync-thread'])
async def test_secondary_timeout_reaches_keyless_rescue(dispatch, monkeypatch, sync):
    release = threading.Event()

    async def hang(urls, format=None):
        await asyncio.sleep(10)

    def sync_hang(urls, format=None):
        release.wait(2)

    secondary = SimpleNamespace(name='secondary', extract=sync_hang if sync else hang)
    monkeypatch.setattr(wt, '_get_secondary_provider', lambda *args: secondary)
    monkeypatch.setattr(wte, '_rescue_eligible', lambda p: True)
    rescue = Mock(return_value=[{'url': URL, 'content': 'rescued'}])
    monkeypatch.setattr('plugins.web.keyless_mcp.extract_with_failover', rescue)
    primary = SimpleNamespace(name='primary', extract=AsyncMock(return_value=[
        {'url': URL, 'error': 'primary outage'},
    ]))
    try:
        result = await dispatch.run(primary)
    finally:
        release.set()
    assert result['results'][0]['content'] == 'rescued'
    rescue.assert_called_once_with('primary', [URL])
    dispatch.put.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('timeout', [0, -1])
async def test_public_tool_can_disable_provider_timeout(dispatch, timeout):
    dispatch.config['extract_timeout'] = timeout

    async def slower_than_cap(urls, format=None):
        await asyncio.sleep(0.04)
        return [{'url': URL, 'content': 'ok'}]

    result = await dispatch.run(SimpleNamespace(name='primary', extract=slower_than_cap))
    assert result['results'][0]['content'] == 'ok'


@pytest.mark.asyncio
async def test_redirected_primary_is_served_and_cached_under_requested_url(dispatch):
    target = 'https://example.com/end'
    row = {'url': target, 'content': 'redirect content', 'title': 'Redirect',
           'metadata': {'sourceURL': URL}}
    result = await dispatch.run(SimpleNamespace(name='primary', extract=AsyncMock(return_value=[row])))
    assert result['results'][0]['content'] == 'redirect content'
    assert result['results'][0]['url'] == target
    dispatch.put.assert_called_once_with(URL, 'redirect content', 'Redirect', format=None, provider='primary')


@pytest.mark.asyncio
async def test_redirect_cache_readback_never_poisons_target_key(dispatch, monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(cache, 'extract_cache_get', dispatch.real_get)
    monkeypatch.setattr(cache, 'extract_cache_put', dispatch.real_put)
    target = 'https://example.com/end'
    provider = SimpleNamespace(name='primary', extract=AsyncMock(return_value=[
        {'url': target, 'content': 'redirect content', 'metadata': {'sourceURL': URL}},
    ]))
    first = await dispatch.run(provider)
    second = await dispatch.run(provider)
    assert first['results'][0]['content'] == second['results'][0]['content'] == 'redirect content'
    provider.extract.assert_awaited_once_with([URL], format=None)
    assert dispatch.real_get(URL, format=None, provider='primary')['content'] == 'redirect content'
    assert dispatch.real_get(target, format=None, provider='primary') is None


@pytest.mark.asyncio
async def test_redirect_source_wins_over_target_with_duplicate_and_out_of_order_rows(dispatch):
    other = 'https://example.com/other'
    rows = [
        {'url': URL, 'content': 'other content', 'metadata': {'sourceURL': other}},
        {'url': other, 'content': 'first content', 'metadata': {'sourceURL': URL}},
        {'url': other, 'content': 'second content', 'metadata': {'sourceURL': URL}},
    ]
    provider = SimpleNamespace(name='primary', extract=AsyncMock(return_value=rows))
    result = await dispatch.run(provider, [URL, other, URL])
    assert [r['content'] for r in result['results']] == ['first content', 'other content', 'second content']
    assert [(c.args[0], c.args[1]) for c in dispatch.put.call_args_list] == [
        (other, 'other content'), (URL, 'first content'), (URL, 'second content'),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', [{}, {'sourceURL': 'https://unknown.example/'}, {'sourceURL': []}, []])
async def test_unassociated_redirect_rows_are_not_positional_or_cached(dispatch, metadata):
    row = {'url': 'https://example.com/unrequested', 'content': 'wrong page', 'metadata': metadata}
    result = await dispatch.run(SimpleNamespace(name='primary', extract=AsyncMock(return_value=[row])))
    assert result['results'][0]['url'] == URL
    assert result['results'][0]['error']
    assert not result['results'][0]['content']
    dispatch.put.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['secondary', 'rescue'])
async def test_redirect_fallback_and_rescue_stay_one_shot_uncached(dispatch, monkeypatch, lane):
    redirected = {'url': 'https://example.com/end', 'content': 'fallback content', 'metadata': {'sourceURL': URL}}
    if lane == 'secondary':
        secondary = SimpleNamespace(name='secondary', extract=AsyncMock(return_value=[redirected]))
        monkeypatch.setattr(wt, '_get_secondary_provider', lambda *a: secondary)
    else:
        monkeypatch.setattr(wte, '_rescue_eligible', lambda p: True)
        monkeypatch.setattr('plugins.web.keyless_mcp.extract_with_failover', lambda *a: [redirected])
    primary = SimpleNamespace(name='primary', extract=AsyncMock(return_value=[{'url': URL, 'error': 'outage'}]))
    result = await dispatch.run(primary)
    assert result['results'][0]['content'] == 'fallback content'
    dispatch.put.assert_not_called()


@pytest.mark.asyncio
async def test_redirect_source_cannot_replace_an_ineligible_policy_occurrence(dispatch, monkeypatch):
    blocked = 'https://example.com/blocked'
    primary = SimpleNamespace(name='primary', extract=AsyncMock(return_value=[
        {'url': blocked, 'error': 'website policy denied', 'blocked_by_policy': True},
        {'url': URL, 'error': 'outage'},
    ]))
    secondary = SimpleNamespace(name='secondary', extract=AsyncMock(return_value=[
        # The target is retried, but the claimed source is policy-blocked.
        {'url': URL, 'content': 'must not replace either', 'metadata': {'sourceURL': blocked}},
    ]))
    monkeypatch.setattr(wt, '_get_secondary_provider', lambda *a: secondary)
    result = await dispatch.run(primary, [blocked, URL])
    assert result['results'][0]['error'] == 'website policy denied'
    assert result['results'][1]['error'] == 'outage'
    secondary.extract.assert_awaited_once_with([URL], format=None)
    dispatch.put.assert_not_called()
