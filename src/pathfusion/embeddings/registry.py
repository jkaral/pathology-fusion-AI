"""Name -> encoder class registry (the names used in configs and feature files)."""

from __future__ import annotations

from pathfusion.embeddings.base import TileEncoder
from pathfusion.embeddings.lunit_dino import LunitDINO
from pathfusion.embeddings.phikon import Phikon
from pathfusion.embeddings.phikonv2 import PhikonV2
from pathfusion.embeddings.uni2h import UNI2h
from pathfusion.embeddings.virchow2 import Virchow2

ENCODERS: dict[str, type[TileEncoder]] = {
    UNI2h.name: UNI2h,
    Virchow2.name: Virchow2,
    PhikonV2.name: PhikonV2,
    # ungated encoders used by the CPU pilot (conf/pilot.yaml)
    Phikon.name: Phikon,
    LunitDINO.name: LunitDINO,
}

EMBED_DIMS: dict[str, int] = {k: v.embed_dim for k, v in ENCODERS.items()}


def get_encoder(name: str) -> TileEncoder:
    try:
        return ENCODERS[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown encoder {name!r}; available: {sorted(ENCODERS)}") from exc
