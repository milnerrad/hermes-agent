"""Pending #95296: literal hashes must not corrupt multiline shell arguments."""
import os

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as scan


def test_hash_inside_word_is_literal_in_multiline_single_quote(tmp_path):
    fifo = tmp_path / "public" / "app.js"
    fifo.parent.mkdir()
    os.mkfifo(fifo)
    assert not scan(f"runner tag#literal 'first line\n{fifo}\nlast line'", cwd=str(tmp_path))


def test_real_comment_hides_referenced_script_path(tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    assert not scan(f"runner ok # sh {script}", cwd=str(tmp_path))


def test_command_after_comment_remains_visible(tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    assert scan(f"runner ok # ignored\nsh {script}", cwd=str(tmp_path))
