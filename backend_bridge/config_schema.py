"""MissionConfig validation (docs/protocol.md §6), option catalog and physics preview."""
from __future__ import annotations

import shutil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from . import _paths  # noqa: F401
from redshift_autonomy import gemma_agent_brain as gab
from redshift_autonomy.agent_fsm import display_name, role_of
from redshift_autonomy.edge_budget import PROFILES
from redshift_autonomy.physics_model import RoverPhysics
from redshift_sim.world_builder import SECTORS, robot_names

SectorId = Literal['small', 'medium', 'large']
EdgeProfileId = Literal['orin_nano_15w', 'orin_nx_25w', 'agx_orin_60w', 'rpi5_8w']
SECTOR_LABELS = {'small': 'Small', 'medium': 'Medium', 'large': 'Large'}
SPEED_CHOICES = (0.5, 1.0, 1.5, 2.0)


class Fleet(BaseModel):
    model_config = ConfigDict(extra='forbid')
    rovers: int = Field(2, ge=1, le=2)
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
    sim_speed: float = Field(1.0, ge=0.25, le=3.0)     # target real-time factor; also changeable live
    # advisory Gemma brain (cloud). Off, or with no API key, fleet_brain_node still runs and answers with the local
    # rules, so every decision keeps the same schema; the local safety layer has authority either way.
    ai_brain: bool = True

    @property
    def robots(self) -> list[str]:
        return robot_names(self.fleet.model_dump())


def gazebo_available() -> bool:
    return shutil.which('gz') is not None


def brain_key_present() -> bool:
    """Whether a Gemini API key is configured (environment or the repo .env). Only the boolean leaves this
    function: the key itself is never returned, logged or sent to a client."""
    try:
        return gab.load_api_key() is not None
    except Exception:  # noqa: BLE001 - an unreadable .env must not break the options endpoint
        return False


def options() -> dict:
    d = MissionConfig()
    return {
        'sectors': [{'id': k, 'label': SECTOR_LABELS[k], 'size_m': L, 'res_m': r, 'cells': int(round(L / r))}
                    for k, (L, r) in SECTORS.items()],
        'gravity': {'min': 1.0, 'max': 3.0, 'step': 0.1, 'default': d.gravity_g},
        'fleet': {'rovers': {'min': 1, 'max': 2, 'default': d.fleet.rovers},
                  'drones': {'min': 0, 'max': 1, 'default': d.fleet.drones},
                  'roles': {r: {'role': role_of(r), 'name': display_name(r)} for r in ('rover_0', 'rover_1', 'scout_0')}},
        'edge_profiles': [{k: v for k, v in p.as_dict().items()} for p in PROFILES.values()],
        'backends': [{'id': 'gazebo', 'label': 'Gazebo Harmonic', 'available': gazebo_available()},
                     {'id': 'mock', 'label': 'Mock physics', 'available': True}],
        'sim_speed': {'choices': list(SPEED_CHOICES), 'min': 0.25, 'max': 3.0, 'default': d.sim_speed},
        'brain': {'model': gab.DEFAULT_MODEL, 'key_present': brain_key_present(), 'default_enabled': d.ai_brain},
        'defaults': d.model_dump(),
    }


def physics_preview(g: float) -> dict:
    g = min(3.0, max(1.0, float(g)))
    return RoverPhysics().limits(g).as_dict()
