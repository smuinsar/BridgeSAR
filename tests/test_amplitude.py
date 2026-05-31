"""Parity tests: reading amplitudes straight from the zarr (AmplitudeStore)
must match the exported-GeoTIFF path, and the store must survive pickling so it
can be sent to joblib workers."""

import pickle

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
xr = pytest.importorskip("xarray")
da = pytest.importorskip("dask.array")
pytest.importorskip("rioxarray")

from bridgesar.amplitude import (
    AmplitudeStore, mean_stack, load_amp_square, export_amplitudes_from_zarr,
)


def _make_amp_zarr(path):
    """Build a small amplitude-only zarr on the anisotropic 5 x 10 m OPERA grid."""
    ny, nx, nt = 12, 9, 4
    rng = np.random.default_rng(0)
    amp = (rng.random((nt, ny, nx)).astype(np.float32) + 0.1)
    amp[:, 0, 0] = 0.0  # exercise the zero -> NaN masking path
    x = 550000 + 5 * np.arange(nx)
    y = 4180000 - 10 * np.arange(ny)
    times = ["20210105", "20210117", "20210129", "20210210"]
    ds = xr.Dataset(
        {"amplitude": (["time", "y", "x"], da.from_array(amp, chunks=(1, ny, nx)))},
        coords={"time": times, "y": y, "x": x})
    ds = ds.rio.write_crs("EPSG:32610").rio.write_transform()
    ds.attrs["amplitude_only"] = True
    ds.to_zarr(str(path), mode="w")
    return times


def test_zarr_matches_geotiff(tmp_path):
    zp = tmp_path / "amp.zarr"
    times = _make_amp_zarr(zp)
    out = tmp_path / "amps"
    export_amplitudes_from_zarr(str(zp), str(out))

    store = AmplitudeStore(str(zp))
    assert store.dates == times

    for d in times:
        az, _, _, dxz = store.load(d)
        ag, _, _, dxg = load_amp_square(str(out / f"{d}_amp.tif"))
        assert dxz == dxg == 5.0
        assert az.shape == (24, 9)  # 12 rows resampled to square 5 m pixels
        np.testing.assert_array_equal(np.nan_to_num(az), np.nan_to_num(ag))

    mz = store.mean_stack(times)[0]
    mg = mean_stack([str(out / f"{d}_amp.tif") for d in times])[0]
    np.testing.assert_array_equal(np.nan_to_num(mz), np.nan_to_num(mg))


def test_amplitude_store_is_picklable(tmp_path):
    zp = tmp_path / "amp.zarr"
    times = _make_amp_zarr(zp)
    store = AmplitudeStore(str(zp))
    store.load(times[0])                      # opens the lazy handle
    restored = pickle.loads(pickle.dumps(store))
    assert restored._amp is None             # handle dropped on pickle
    a = restored.load(times[0])[0]           # reopens lazily in this process
    assert a.shape == (24, 9)


def test_rejects_non_amplitude_zarr(tmp_path):
    zp = tmp_path / "plain.zarr"
    xr.Dataset({"slc": (["y", "x"], np.zeros((3, 3)))}).to_zarr(str(zp), mode="w")
    with pytest.raises(ValueError):
        AmplitudeStore(str(zp))
