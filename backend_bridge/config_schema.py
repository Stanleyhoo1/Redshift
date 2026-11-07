"""MissionConfig validation (docs/protocol.md §6), option catalog and physics preview."""
from __future__ import annotations

import shutil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from . import _paths  # noqa: F401
from exoswarm_autonomy.edge_budget import PROFILES
from exoswarm_autonomy.physics_model import RoverPhysics
from exoswarm_sim.world_builder import SECTORS, robot_names

SectorId = Literal['small', 'medium', 'large']
EdgeProfileId = Literal['orin_nano_15w', 'orin_nx_25w', 'agx_orin_60w', 'rpi5_8w']
SECTOR_LABELS = {'small': 'Small', 'medium': 'Medium', 'large': 'Large'}


class Fleet(BaseModel):
    model_config = ConfigDict(extra='forbid')
    rovers: int = Field(1, ge=1, le=2)
    drones: int = Field(1, ge=0, le=1)


class MissionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sector: SectorId = 'medium'
    gravity_g: float = Field(2.2, ge=1.0, le=3.0)
    fleet: Fleet = Field(default_factory=Fleet)
    edge_profile: EdgeProfileId = 'orin_nano_15w'
    sim_backend: Literal['gazebo', 'mock'] = 'gazebo'
    seed: int = Field(42, ge=0, le=2**31 - 1)
    gazebo_gui: bool = False

    @property
    def robots(self) -> list[str]:
        return robot_names(self.fleet.model_dump())


def gazebo_available() -> bool:
    return shutil.which('gz') is not None


def options() -> dict:
    d = MissionConfig()
    return {
        'sectors': [{'id': k, 'label': SECTOR_LABELS[k], 'size_m': L, 'res_m': r, 'cells': int(round(L / r))}
                    for k, (L, r) in SECTORS.items()],
        'gravity': {'min': 1.0, 'max': 3.0, 'step': 0.1, 'default': d.gravity_g},
        'fleet': {'rovers': {'min': 1, 'max': 2, 'default': 1}, 'drones': {'min': 0, 'max': 1, 'default': 1}},
        'edge_profiles': [{k: v for k, v in p.as_dict().items()} for p in PROFILES.values()],
        'backends': [{'id': 'gazebo', 'label': 'Gazebo Harmonic', 'available': gazebo_available()},
                     {'id': 'mock', 'label': 'Mock physics', 'available': True}],
        'defaults': d.model_dump(),
    }


def physics_preview(g: float) -> dict:
    g = min(3.0, max(1.0, float(g)))
    return RoverPhysics().limits(g).as_dict()
