import pytest
from pydantic import ValidationError

from backend_bridge.config_schema import MissionConfig, options, physics_preview


def test_defaults_and_robots():
    c = MissionConfig()
    assert c.sector == 'medium' and c.gravity_g == 2.2 and c.edge_profile == 'orin_nano_15w'
    assert c.robots == ['rover_0', 'rover_1', 'scout_0']        # default fleet: 2 rovers + 1 drone
    assert c.sim_speed == 1.0 and MissionConfig(sim_speed=2.0).sim_speed == 2.0
    assert MissionConfig(fleet={'rovers': 2, 'drones': 0}).robots == ['rover_0', 'rover_1']


@pytest.mark.parametrize('bad', [{'gravity_g': 3.5}, {'gravity_g': 0.9}, {'fleet': {'rovers': 3, 'drones': 1}},
                                 {'fleet': {'rovers': 0, 'drones': 1}}, {'sector': 'huge'},
                                 {'edge_profile': 'gpu_9000'}, {'unknown_key': 1}, {'sim_speed': 0.1},
                                 {'sim_speed': 5.0}, {'ai_brain': 'maybe'}, {'ai_brain': 2}, {'ai_brain': None}])
def test_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        MissionConfig(**bad)


def test_options_catalog_shape():
    o = options()
    assert [s['id'] for s in o['sectors']] == ['small', 'medium', 'large']
    assert o['sectors'][1]['cells'] == 200
    assert {p['id'] for p in o['edge_profiles']} == {'orin_nano_15w', 'orin_nx_25w', 'agx_orin_60w', 'rpi5_8w'}
    assert {'id', 'label', 'device', 'watts_idle', 'watts_max', 'perf_scale', 'perception_hz_max',
            'max_expansions'} <= set(o['edge_profiles'][0])
    assert o['defaults']['fleet'] == {'rovers': 2, 'drones': 1}
    assert o['fleet']['roles']['rover_1'] == {'role': 'recovery', 'name': 'Heavy Recovery Rover 2'}
    assert any(b['id'] == 'mock' and b['available'] for b in o['backends'])
    assert o['sim_speed']['choices'] == [0.5, 1.0, 1.5, 2.0] and o['sim_speed']['default'] == 1.0


def test_physics_preview_matches_model_and_clamps():
    p = physics_preview(2.2)
    assert p['factor'] == 'drive_torque' and 15.5 < p['max_slope_deg'] < 18.0
    assert set(p) >= {'g', 'max_slope_deg', 'factor', 'drive_deg', 'slip_deg', 'tip_deg', 'tip_alarm_deg', 'v_max',
                      'energy_j_per_m'}
    assert physics_preview(9.0)['g'] == 3.0


def test_ai_brain_flag_defaults_on_and_reaches_the_runtime_config():
    assert MissionConfig().ai_brain is True
    assert MissionConfig(ai_brain=False).ai_brain is False
    assert MissionConfig(ai_brain=False).model_dump()['ai_brain'] is False     # mission_manager writes model_dump()
    assert MissionConfig.model_validate_json('{"ai_brain": false}').ai_brain is False


FAKE_KEY = 'AIzaFAKE-unit-test-key-0123456789'


def test_options_brain_block_reports_key_presence_without_the_key(monkeypatch):
    import json

    from redshift_autonomy import gemma_agent_brain as gab
    monkeypatch.setenv(gab.API_KEY_ENV, FAKE_KEY)
    o = options()
    assert o['brain'] == {'model': 'gemma-4-26b-a4b-it', 'key_present': True, 'default_enabled': True}
    assert FAKE_KEY not in json.dumps(o)
    assert o['defaults']['ai_brain'] is True
    monkeypatch.setattr(gab, 'load_api_key', lambda env_path=None: None)
    assert options()['brain']['key_present'] is False

    def broken(env_path=None):
        raise PermissionError('.env unreadable')
    monkeypatch.setattr(gab, 'load_api_key', broken)
    assert options()['brain']['key_present'] is False
