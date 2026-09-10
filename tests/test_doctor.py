"""Installation diagnostics identify their source and separate quick checks."""

import json
import sys
from pathlib import Path

from nro.bin import doctor


def _record(tmp_path):
    return {
        "mode": "branch",
        "branch": "dev",
        "environment": str(tmp_path / "environment"),
    }


def test_doctor_reports_selected_installation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doctor.site, "CHECKOUT", tmp_path / "checkout")
    monkeypatch.setattr(doctor.site, "installation_record", lambda: _record(tmp_path))
    monkeypatch.setattr(doctor.site, "site_file", lambda: tmp_path / "site.toml")
    calls = []
    monkeypatch.setattr(
        doctor,
        "check_installation",
        lambda **options: calls.append(options) or [],
    )

    doctor.main([])

    output = capsys.readouterr().out
    assert f"SELECTED checkout: {tmp_path / 'checkout'}" in output
    assert "SELECTED branch: dev" in output
    assert f"SELECTED executable: {sys.executable}" in output
    assert calls == [{"deep": False, "with_oslom": True, "slurm": True, "quick": True}]


def test_deep_doctor_uses_full_checks_and_keeps_json_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doctor.site, "CHECKOUT", tmp_path / "checkout")
    monkeypatch.setattr(doctor.site, "installation_record", lambda: _record(tmp_path))
    monkeypatch.setattr(doctor.site, "site_file", lambda: tmp_path / "site.toml")
    calls = []
    monkeypatch.setattr(
        doctor,
        "check_installation",
        lambda **options: calls.append(options) or [],
    )

    doctor.main(["--deep", "--json"])

    output = json.loads(capsys.readouterr().out)
    assert output[0]["category"] == "installation"
    assert calls == [{"deep": True, "with_oslom": True, "slurm": True, "quick": False}]


def test_shared_installation_is_reported_as_main(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor.site, "CHECKOUT", Path("/srv/nro"))
    monkeypatch.setattr(
        doctor.site,
        "installation_record",
        lambda: {"mode": "shared", "environment": "/srv/nro/.nro-env"},
    )
    monkeypatch.setattr(doctor.site, "site_file", lambda: Path("/srv/nro/site.toml"))

    details = {row["name"]: row["detail"] for row in doctor.installation_details()}

    assert details["branch"] == "main"
