import os
from pathlib import Path

from nro.orchestration.runtime import (
    CONFIGURATION_FINGERPRINT_ENV,
    select_runtime_config,
    selected_configuration_fingerprint,
)
from nro.configuration.store import ConfigStore


def test_direct_workflow_selection_exports_intrinsic_configuration_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("NRO_RUNTIME_CONFIG", raising=False)
    monkeypatch.delenv(CONFIGURATION_FINGERPRINT_ENV, raising=False)
    expected = ConfigStore().resolve("main").configuration("clean").fingerprint

    runtime = select_runtime_config(
        project="demo",
        workflow_id="main",
        derivative_class="clean",
        bids_root=tmp_path / "bids",
    )

    assert runtime.is_file()
    assert os.environ["NRO_RUNTIME_CONFIG"] == str(runtime)
    assert selected_configuration_fingerprint() == expected
