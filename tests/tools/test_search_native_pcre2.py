"""Custom pattern protections must use stable's shared native/shell rg transport."""
import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = [pytest.mark.platforms('posix'),
              pytest.mark.skipif(shutil.which('rg') is None, reason='requires ripgrep')]


@pytest.mark.parametrize('pattern, expected', [
    (r'alpha (?=foo)', 2), (r'alpha (foo) \1 omega', 1),
    (r'--glob=\*\.secret', 1), ('alpha', 2),
])
@pytest.mark.parametrize('native', [True, False], ids=['native', 'shell'])
def test_public_search_preserves_pattern_protection_on_both_transports(tmp_path, monkeypatch, pattern, expected, native):
    monkeypatch.setenv('HERMES_NATIVE_FILE_READ', '1' if native else '0')
    (tmp_path / '-root.txt').write_text('alpha foo omega\nalpha foo foo omega\n--glob=*.secret\n')
    env = LocalEnvironment(cwd=str(tmp_path))
    ops = ShellFileOperations(env, cwd=str(tmp_path))
    ops._rg_resolution_cache['rg'] = shutil.which('rg')
    shell_calls = []
    real = ops._exec

    def record(command, **kwargs):
        shell_calls.append(command)
        return real(command, **kwargs)

    monkeypatch.setattr(ops, '_exec', record)
    result = ops.search(pattern, path='-root.txt')
    assert result.error is None, result.to_dict()
    assert result.total_count == expected
    if native:
        assert shell_calls == [], shell_calls
    else:
        assert any('pipefail' in command for command in shell_calls)
