"""BridgeSAR — geometry-guided bridge-water clearance from SAR multipath stripes.

Public API
----------
>>> from bridgesar import BridgeConfig, HyperParams, ClearancePipeline
>>> cfg = BridgeConfig.from_yaml("configs/baybridge_p115.yaml")
>>> pipe = ClearancePipeline(cfg)
>>> pipe.setup()                 # load zarr LOS geometry + OSM + global S-bounce
>>> df = pipe.run_timeseries()   # per-date clearance estimates
>>> kept = pipe.filter_timeseries(df)
"""

from .config import BridgeConfig, HyperParams, OSMQuery, WLStation
from .clearance import ClearancePipeline
from .amplitude import AmplitudeStore

__version__ = "0.1.0"

__all__ = [
    "BridgeConfig",
    "HyperParams",
    "OSMQuery",
    "WLStation",
    "ClearancePipeline",
    "AmplitudeStore",
    "__version__",
]
