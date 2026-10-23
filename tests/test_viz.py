from __future__ import annotations

import gzip
import json
import os
import threading
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from nro.viz import data, service
from nro.viz.catalog import Catalogs, build_layers
from nro.viz.server import make_server

VERTICES = {"L": 6, "R": 5}
MASKS = {
    "L": np.array([True, True, False, True, True, True]),
    "R": np.array([True, False, True, True, True]),
}


def _brain_models() -> nib.cifti2.BrainModelAxis:
    left = nib.cifti2.BrainModelAxis.from_mask(MASKS["L"], name="CortexLeft")
    right = nib.cifti2.BrainModelAxis.from_mask(MASKS["R"], name="CortexRight")
    return left + right


def _cifti(path: Path, values: np.ndarray, first_axis) -> Path:
    header = nib.cifti2.Cifti2Header.from_axes((first_axis, _brain_models()))
    nib.save(nib.cifti2.Cifti2Image(values.astype(np.float32), header), str(path))
    return path


def _metric(path: Path, values: np.ndarray) -> Path:
    image = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(values.astype(np.float32), intent="NIFTI_INTENT_SHAPE")]
    )
    nib.save(image, str(path))
    return path


def _columns() -> int:
    return int(MASKS["L"].sum() + MASKS["R"].sum())


def test_layers_pair_hemispheres_and_classify_cifti(tmp_path: Path) -> None:
    left = _metric(tmp_path / "sub-01_space-fsnative_hemi-L_thickness.shape.gii", np.arange(6))
    right = _metric(tmp_path / "sub-01_space-fsnative_hemi-R_thickness.shape.gii", np.arange(5))
    scalars = _cifti(
        tmp_path / "sub-01_space-fsnative_desc-networks_probability.dscalar.nii",
        np.ones((2, _columns())),
        nib.cifti2.ScalarAxis(["language", "default"]),
    )
    series = _cifti(
        tmp_path / "sub-01_space-fsnative_desc-dynamicConnectivity_bold.dtseries.nii",
        np.ones((4, _columns())),
        nib.cifti2.SeriesAxis(0, 1, 4),
    )
    volume = tmp_path / "sub-01_desc-preproc_T1w.nii.gz"

    layers = build_layers(
        [("anat", left, "fsnative"), ("anat", right, "fsnative"), ("anat", volume, None)]
        + [("networks", scalars, "fsnative"), ("dynconn", series, "fsnative")],
        "01",
    )

    by_label = {layer.label: layer for layer in layers}
    assert set(by_label) == {
        "thickness",
        "desc-networks probability",
        "desc-dynamicConnectivity bold",
    }
    assert by_label["thickness"].paths == {"L": left, "R": right}
    assert by_label["thickness"].vertex_counts == VERTICES
    assert by_label["desc-networks probability"].maps == ("language", "default")
    assert by_label["desc-networks probability"].vertex_counts == VERTICES
    assert by_label["desc-dynamicConnectivity bold"].kind == "connectivity"


def test_cifti_maps_fill_unaddressed_vertices_with_nan(tmp_path: Path) -> None:
    values = np.arange(2 * _columns(), dtype=float).reshape(2, _columns())
    path = _cifti(
        tmp_path / "sub-01_space-fsnative_desc-test_map.dscalar.nii",
        values,
        nib.cifti2.ScalarAxis(["first", "second"]),
    )
    (layer,) = build_layers([("networks", path, "fsnative")], "01")

    hemispheres = data.layer_map(layer, 1)

    left_columns = int(MASKS["L"].sum())
    expected_left = np.full(6, np.nan)
    expected_left[MASKS["L"]] = values[1, :left_columns]
    expected_right = np.full(5, np.nan)
    expected_right[MASKS["R"]] = values[1, left_columns:]
    np.testing.assert_array_equal(hemispheres["L"], expected_left)
    np.testing.assert_array_equal(hemispheres["R"], expected_right)
    with pytest.raises(IndexError):
        data.layer_map(layer, 2)


def test_seed_correlation_matches_pearson_correlation(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    series = rng.normal(size=(40, _columns()))
    series[:, -1] = 2.0
    path = _cifti(
        tmp_path / "sub-01_space-fsnative_desc-dynamicConnectivity_bold.dtseries.nii",
        series,
        nib.cifti2.SeriesAxis(0, 1, 40),
    )
    (layer,) = build_layers([("dynconn", path, "fsnative")], "01")

    hemispheres = data.SeedCorrelation().correlate(layer, "L", 3)

    seed_column = 2  # Left vertex 3 is the third addressed left vertex.
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.array(
            [
                np.corrcoef(series[:, seed_column], series[:, column])[0, 1]
                for column in range(_columns())
            ]
        )
    left_columns = int(MASKS["L"].sum())
    np.testing.assert_allclose(hemispheres["L"][MASKS["L"]], expected[:left_columns], atol=1e-6)
    np.testing.assert_allclose(
        hemispheres["R"][MASKS["R"]][:-1], expected[left_columns:-1], atol=1e-6
    )
    assert np.isnan(hemispheres["R"][4])
    assert np.isnan(hemispheres["L"][2])
    with pytest.raises(ValueError, match="no data"):
        data.SeedCorrelation().correlate(layer, "L", 2)


def _catalogs(rows_by_branch: dict[str, list[dict]], *layers) -> Catalogs:
    catalogs = Catalogs(lambda branch: rows_by_branch.get(branch, []))
    for layer in layers:
        catalogs.get("")._layers[layer.id] = layer
    return catalogs


@pytest.fixture()
def served(tmp_path: Path):
    left = _metric(tmp_path / "sub-01_space-fsnative_hemi-L_sulc.shape.gii", np.arange(6))
    right = _metric(tmp_path / "sub-01_space-fsnative_hemi-R_sulc.shape.gii", -np.arange(5))
    (layer,) = build_layers([("anat", left, "fsnative"), ("anat", right, "fsnative")], "01")
    server = make_server(_catalogs({}, layer), host="127.0.0.1", port=0, token="secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", layer
    finally:
        server.shutdown()
        server.server_close()


def _error(opener, url: str) -> tuple[int, dict]:
    with pytest.raises(urllib.error.HTTPError) as caught:
        opener.open(url)
    return caught.value.code, json.loads(caught.value.read())


def test_server_requires_token_before_serving_data(served) -> None:
    base, layer = served
    anonymous = urllib.request.build_opener()

    status, body = _error(anonymous, f"{base}/api/layer?layer={layer.id}")
    assert status == 403 and "token" in body["error"]
    status, _body = _error(anonymous, f"{base}/?token=wrong")
    assert status == 403
    request = urllib.request.Request(f"{base}/api/ping", headers={"Authorization": "Bearer secret"})
    with anonymous.open(request) as response:
        assert json.loads(response.read()) == {"ok": True}


def test_server_returns_layer_values_after_login(served) -> None:
    base, layer = served
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    with opener.open(f"{base}/?token=secret&project=demo") as response:
        assert response.url == f"{base}/?project=demo"
        assert b"nro viz" in response.read()

    request = urllib.request.Request(
        f"{base}/api/layer?layer={layer.id}", headers={"Accept-Encoding": "gzip"}
    )
    with opener.open(request) as response:
        counts = json.loads(response.headers["X-Counts"])
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    values = np.frombuffer(body, dtype="<f4")
    assert counts == VERTICES
    np.testing.assert_array_equal(values, np.concatenate([np.arange(6), -np.arange(5)]))

    status, body = _error(opener, f"{base}/api/layer?layer=missing")
    assert status == 400
    status, body = _error(opener, f"{base}/static/../catalog.py")
    assert status == 404


def test_server_keeps_branch_catalogs_separate_and_refreshes_one() -> None:
    rows = {
        "": [{"project": "demo", "participant": "01", "module": "anat", "status": "Success"}],
        "main": [],
    }
    restarts = []
    catalogs = _catalogs(rows)
    server = make_server(
        catalogs, host="127.0.0.1", port=0, token="secret", restart=lambda: restarts.append(1)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        opener.open(f"{base}/?token=secret").read()
        with opener.open(f"{base}/api/participants?branch=main") as response:
            assert json.loads(response.read()) == []
        with opener.open(f"{base}/api/participants") as response:
            assert json.loads(response.read())[0]["participant"] == "01"

        rows["main"] = [
            {"project": "demo", "participant": "02", "module": "anat", "status": "Success"}
        ]
        request = urllib.request.Request(f"{base}/api/refresh?branch=main", data=b"", method="POST")
        with opener.open(request) as response:
            assert json.loads(response.read()) == {"participants": 1}
        assert catalogs.get("").participants()[0]["participant"] == "01"

        anonymous = urllib.request.build_opener()
        with pytest.raises(urllib.error.HTTPError) as caught:
            anonymous.open(urllib.request.Request(f"{base}/api/restart", data=b"", method="POST"))
        assert caught.value.code == 403 and not restarts
        request = urllib.request.Request(f"{base}/api/restart", data=b"", method="POST")
        with opener.open(request) as response:
            assert json.loads(response.read()) == {"restarting": True}
        assert restarts == [1]
    finally:
        server.shutdown()
        server.server_close()


def test_seed_cache_keeps_series_within_budget(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        series = np.random.default_rng(index).normal(size=(10, _columns()))
        paths.append(
            _cifti(
                tmp_path / f"sub-0{index}_desc-dynamicConnectivity_bold.dtseries.nii",
                series,
                nib.cifti2.SeriesAxis(0, 1, 10),
            )
        )
    one_series = 10 * _columns() * 4
    cache = data.SeedCorrelation(budget_bytes=2 * one_series)

    for path in (paths[0], paths[1], paths[0], paths[2]):
        cache.prepare(path)

    assert cache.resident() == (paths[0], paths[2])


def test_state_is_private_per_user_and_checkout(tmp_path: Path) -> None:
    first = service.state_directory(tmp_path, Path("/checkouts/main"))
    second = service.state_directory(tmp_path, Path("/checkouts/dev"))

    assert first.parent == tmp_path / "viewers" / f"uid-{os.getuid()}"
    assert first != second
    assert first.stat().st_mode & 0o777 == 0o700
    token = service.access_token(first)
    assert service.access_token(first) == token
    assert (first / "token").stat().st_mode & 0o777 == 0o600


def test_server_prefers_stable_port_and_moves_when_taken(tmp_path: Path, monkeypatch) -> None:
    import socket

    preferred_port = service.preferred_port
    holder = socket.socket()
    holder.bind(("0.0.0.0", 0))
    holder.listen()
    taken = holder.getsockname()[1]
    monkeypatch.setattr(service, "preferred_port", lambda uid=None: taken)
    monkeypatch.setattr("nro.viz.catalog.branch_rows", lambda branch: [])
    try:
        server = service.bind(tmp_path, "secret", port=None, cache_bytes=1, restart=None)
        try:
            assert server.server_address[1] != taken
            assert taken < server.server_address[1] < taken + service.PORT_ATTEMPTS
        finally:
            server.server_close()
    finally:
        holder.close()
    assert 20000 <= preferred_port(12345) < 60000
    assert preferred_port(12345) == preferred_port(12345)


def test_submit_queues_a_batch_job_on_the_viewing_partition(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="4242;cluster\n", stderr="")

    monkeypatch.setattr(service.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(service.subprocess, "run", run)
    monkeypatch.setenv("SLURM_JOB_ID", "999")

    job_id = service.submit(
        tmp_path,
        {
            "viewer_cpus": 2,
            "viewer_memory": 32,
            "viewer_time": 12,
            "viewing_partition": "john",
            "account": "nlp",
        },
    )

    command, kwargs = calls[0]
    assert job_id == "4242"
    assert command[:7] == [
        "/usr/bin/sbatch",
        "--parsable",
        "--job-name=nro-viz",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=32G",
        "--time=12:00:00",
    ]
    assert "--partition=john" in command and "--account=nlp" in command
    wrap = next(part for part in command if part.startswith("--wrap="))
    assert "nro.viz.service --serve --state-dir" in wrap
    assert "SLURM_JOB_ID" not in kwargs["env"]
    assert json.loads((tmp_path / "job.json").read_text())["job_id"] == "4242"


def test_viz_reuses_a_running_server_without_submitting(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import nro.bin.viz as viz

    monkeypatch.setattr(viz, "_state", lambda: tmp_path)
    monkeypatch.setattr(
        service, "active_server", lambda root, token: {"host": "node1", "port": 23456}
    )
    monkeypatch.setattr(service, "submit", lambda *args: pytest.fail("must not submit"))
    monkeypatch.setattr(viz.getpass, "getuser", lambda: "user")

    viz.main(["--tunnel-host", "login"])

    output = capsys.readouterr().out
    assert "ssh -N -L 23456:node1:23456 user@login" in output
    assert f"http://localhost:23456/?token={service.access_token(tmp_path)}\n" in output

    viz.main(["--tunnel-host", "login", "-P", "climb", "-p", "c001", "--branch", "main"])

    assert "&branch=main&project=climb&participant=c001" in capsys.readouterr().out


def test_viz_returns_after_queueing_with_no_wait(tmp_path: Path, monkeypatch, capsys) -> None:
    import nro.bin.viz as viz
    import nro.site.configuration as configuration

    monkeypatch.setattr(viz, "_state", lambda: tmp_path)
    monkeypatch.setattr(service, "active_server", lambda root, token: None)
    monkeypatch.setattr(service, "queued_job", lambda root: None)
    monkeypatch.setattr(service, "submit", lambda root, values: "77")
    monkeypatch.setattr(configuration, "settings", lambda: ({},))

    viz.main(["--no-wait"])

    captured = capsys.readouterr()
    assert "Submitted viewer server job 77" in captured.err
    assert "Run `nro viz` again" in captured.out


def test_restart_command_reuses_port_and_state(tmp_path: Path) -> None:
    command = service.restart_argv(tmp_path, 4321, 32)

    assert command[1:] == [
        "-m",
        "nro.viz.service",
        "--serve",
        "--state-dir",
        str(tmp_path),
        "--port",
        "4321",
        "--memory-gb",
        "32",
    ]


def test_open_file_pairs_hemispheres_and_stays_inside_project(tmp_path: Path) -> None:
    project = tmp_path / "BIDS" / "demo"
    maps = project / "derivatives" / "diagnostics" / "sub-01"
    maps.mkdir(parents=True)
    left = _metric(maps / "sub-01_space-fsnative_hemi-L_desc-test_statmap.func.gii", np.arange(6))
    _metric(maps / "sub-01_space-fsnative_hemi-R_desc-test_statmap.func.gii", np.arange(5))
    outside = _metric(tmp_path / "sub-01_space-fsnative_hemi-L_secret.func.gii", np.arange(6))
    (maps / "link_hemi-L_x.func.gii").symlink_to(outside)
    catalogs = Catalogs(lambda branch: [], project_root=lambda name: tmp_path / "BIDS" / name)

    layer = catalogs.open_file("demo", str(left.relative_to(project)))

    assert layer.vertex_counts == VERTICES
    assert catalogs.layer(layer.id) is layer
    with pytest.raises(PermissionError):
        catalogs.open_file("demo", "../sub-01_space-fsnative_hemi-L_secret.func.gii")
    with pytest.raises(PermissionError):
        catalogs.open_file("demo", "derivatives/diagnostics/sub-01/link_hemi-L_x.func.gii")
    with pytest.raises(FileNotFoundError):
        catalogs.open_file("demo", "derivatives/missing.func.gii")


def test_file_endpoint_reports_status_codes(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    project.mkdir()
    _metric(project / "sub-01_space-fsnative_hemi-L_x.func.gii", np.arange(6))
    catalogs = Catalogs(lambda branch: [], project_root=lambda name: tmp_path / name)
    server = make_server(catalogs, host="127.0.0.1", port=0, token="secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/file?project=demo&path="
    opener = urllib.request.build_opener()

    def get(path: str):
        request = urllib.request.Request(base + path, headers={"Authorization": "Bearer secret"})
        try:
            with opener.open(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        status, body = get("sub-01_space-fsnative_hemi-L_x.func.gii")
        assert status == 200 and body["vertex_counts"] == {"L": 6}
        assert get("../etc")[0] == 403
        assert get("nothing.func.gii")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
