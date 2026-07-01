"""
tests/test_env_loading.py

Regression test for a real bug found during manual testing: settings.py
was reading os.getenv() everywhere but never called load_dotenv(), so a
populated .env file was silently ignored and every setting fell back to
its default (e.g. MYSQL_PASSWORD defaulting to empty even when the .env
file had a real password in it).

This test runs config/settings.py in a fresh subprocess with a temp .env
file in place, which is the only reliable way to test module-level
load_dotenv() behavior (it runs once at import time, so re-importing the
already-cached module in-process wouldn't re-trigger it).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_env_file_values_are_actually_loaded(tmp_path):
    project_root = Path(__file__).resolve().parent.parent

    env_file = tmp_path / ".env"
    env_file.write_text(
        "MYSQL_HOST=some-test-host\n"
        "MYSQL_USER=some-test-user\n"
        "MYSQL_PASSWORD=some-test-password\n"
        "MYSQL_DATABASE=some-test-db\n"
    )

    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(project_root)!r})

        # Make BASE_DIR (project_root/parent-of-config) resolve to tmp_path
        # by monkeypatching load_dotenv's target before settings imports it.
        import dotenv
        _real_load_dotenv = dotenv.load_dotenv
        dotenv.load_dotenv = lambda *a, **k: _real_load_dotenv({str(env_file)!r})

        from config.settings import settings as s
        assert s.mysql.host == "some-test-host", s.mysql.host
        assert s.mysql.user == "some-test-user", s.mysql.user
        assert s.mysql.password == "some-test-password", s.mysql.password
        assert s.mysql.database == "some-test-db", s.mysql.database
        print("OK")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout