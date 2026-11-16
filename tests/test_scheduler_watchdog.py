import io
import re
import threading

from nro.orchestration import scheduler_service
from nro.orchestration.registry import SessionMutex


def test_session_mutex_records_outermost_holder():
    mutex = SessionMutex()
    assert mutex.held_since is None
    mutex.acquire()
    first = mutex.held_since
    mutex.acquire()
    assert mutex.held_since == first
    assert mutex.holder == threading.current_thread().name
    mutex.release()
    assert mutex.held_since == first
    mutex.release()
    assert mutex.held_since is None and mutex.holder is None


def test_session_mutex_timeout_does_not_record_failed_acquire():
    mutex = SessionMutex()
    mutex.acquire()
    acquired = []
    thread = threading.Thread(target=lambda: acquired.append(mutex.acquire(timeout=0.01)))
    thread.start()
    thread.join()
    assert acquired == [False]
    assert mutex.holder == threading.current_thread().name
    mutex.release()


def _watchdog(mutex=None):
    exits = []
    watchdog = scheduler_service._StallWatchdog(
        mutex or SessionMutex(),
        dump_after=10.0,
        exit_after=100.0,
        exit_process=exits.append,
    )
    return watchdog, exits


def test_watchdog_ignores_progressing_service(monkeypatch):
    dumps = []
    monkeypatch.setattr(
        scheduler_service.faulthandler, "dump_traceback", lambda **_: dumps.append(1)
    )
    watchdog, exits = _watchdog()
    watchdog.progress = 1000.0
    assert watchdog.check(now=1005.0) is False
    assert dumps == [] and exits == []


def test_watchdog_dumps_periodically_then_exits_on_background_stall(monkeypatch, capsys):
    dumps = []
    monkeypatch.setattr(
        scheduler_service.faulthandler, "dump_traceback", lambda **_: dumps.append(1)
    )
    mutex = SessionMutex()
    mutex.acquire()
    watchdog, exits = _watchdog(mutex)
    watchdog.progress = mutex.held_since + 1e6
    start = mutex.held_since
    assert watchdog.check(now=start + 20.0) is False
    assert watchdog.check(now=start + 25.0) is False
    assert dumps == [1] and exits == []
    assert watchdog.check(now=start + 30.0) is False
    assert dumps == [1, 1] and exits == []
    assert "registry session lock held by MainThread" in capsys.readouterr().err
    assert watchdog.check(now=start + 100.0) is True
    assert exits == [scheduler_service.SCHEDULER_STALL_EXIT_CODE]
    mutex.release()


def test_watchdog_never_exits_inside_a_request(monkeypatch, capsys):
    dumps = []
    monkeypatch.setattr(
        scheduler_service.faulthandler, "dump_traceback", lambda **_: dumps.append(1)
    )
    mutex = SessionMutex()
    holder = threading.Thread(target=mutex.acquire, name="scheduler-command_0")
    holder.start()
    holder.join()
    watchdog, exits = _watchdog(mutex)
    start = mutex.held_since
    watchdog.progress = start
    for elapsed in (15.0, 20.0, 26.0, 500.0):
        assert watchdog.check(now=start + elapsed) is False
    assert exits == []
    assert dumps == [1, 1, 1]
    assert "held by scheduler-command_0" in capsys.readouterr().err


def test_watchdog_rearms_after_recovery(monkeypatch):
    dumps = []
    monkeypatch.setattr(
        scheduler_service.faulthandler, "dump_traceback", lambda **_: dumps.append(1)
    )
    watchdog, exits = _watchdog()
    watchdog.progress = 0.0
    watchdog.check(now=20.0)
    watchdog.progress = 20.0
    watchdog.check(now=21.0)
    watchdog.check(now=40.0)
    assert dumps == [1, 1] and exits == []


def test_timestamped_stream_prefixes_each_line_once():
    target = io.StringIO()
    stream = scheduler_service._TimestampedStream(target)
    stream.write("first ")
    stream.write("line\nsecond line\n")
    lines = target.getvalue().splitlines()
    stamp = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+\+00:00 "
    assert re.fullmatch(stamp + "first line", lines[0])
    assert re.fullmatch(stamp + "second line", lines[1])
