import logging

import pytest

from nro.modules.func.__main__ import _run_with_error_logging


def test_func_entry_point_logs_system_exit_as_error(caplog) -> None:
    def fail(_argv):
        raise SystemExit("specific fatal message")

    with caplog.at_level(logging.ERROR, logger="preprocess"):
        with pytest.raises(SystemExit) as error:
            _run_with_error_logging(fail, [])

    assert error.value.code == 1
    assert "ERROR" in caplog.text
    assert "FATAL: specific fatal message" in caplog.text
