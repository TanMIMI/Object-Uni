import os

os.environ.setdefault("ROPE_TYPE", "3D")

from .infer import SpatialUnifyInference

__all__ = ["SpatialUnifyInference"]
