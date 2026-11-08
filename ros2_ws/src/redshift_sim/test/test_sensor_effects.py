import math

import cv2
import numpy as np
import pytest

pytest.importorskip('rclpy')          # the helpers live in the node module, which imports rclpy

from redshift_autonomy.vision import ImageQualityBaseline, image_quality  # noqa: E402
from redshift_sim.sensor_effects_node import (DUST_RGB, PARTICLES, RAMP_S, STORM_DEFAULTS, STORM_KIND,  # noqa: E402
                                              STORM_RGB, Atmosphere, anomaly_params, degrade_rgb, haze_tint)

GREY = np.full((4, 4, 3), 128.0, np.float32)


def _hue_deg(rgb) -> float:
    px = np.uint8([[np.clip(np.round(rgb), 0, 255)]])
    return float(cv2.cvtColor(px, cv2.COLOR_RGB2HSV_FULL)[0, 0, 0]) * 360.0 / 256.0


def _scene(seed=0, w=320, h=240) -> np.ndarray:
    """A textured regolith frame: smooth albedo variation, a few dark boulders, a lighter sky band."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 6.0)
    base = (base - base.min()) / (base.max() - base.min())
    img = np.stack([150 + 60 * base, 95 + 40 * base, 68 + 30 * base], axis=-1)
    img[: h // 4] = (215, 175, 150)
    for _ in range(8):
        cx, cy, r = int(rng.integers(20, w - 20)), int(rng.integers(h // 3, h - 10)), int(rng.integers(6, 18))
        cv2.ellipse(img, (cx, cy), (r, int(0.7 * r)), 0, 0, 360, (70, 64, 60), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- RGB haze
def test_plain_dust_haze_is_unchanged():
    out = haze_tint(GREY, 0.85)
    assert out == pytest.approx(GREY * (1 - 0.65 * 0.85) + DUST_RGB * (0.65 * 0.85))
    assert haze_tint(GREY, 0.85, 'dust') == pytest.approx(out)
    assert haze_tint(GREY, 0.85, 'unknown_kind') == pytest.approx(out)
    assert haze_tint(GREY, 0.0, STORM_KIND) == pytest.approx(GREY)


def test_storm_haze_is_orange_red_and_stronger_than_dust():
    storm = haze_tint(GREY, 1.0, STORM_KIND)[0, 0]
    dust = haze_tint(GREY, 1.0)[0, 0]
    assert tuple(STORM_RGB) == (205, 88, 48)
    assert storm == pytest.approx(GREY[0, 0] * 0.25 + STORM_RGB * 0.75)          # 0.75·L toward STORM_RGB
    r, g, b = storm
    assert r > 150 and r - g > 50 and g > b                                       # orange-red
    assert _hue_deg(storm) < 20.0 and _hue_deg(storm) < _hue_deg(dust)            # redder than the brown dust
    # stronger: less of the scene's contrast survives (0.25 vs 0.35 of it) and the cast is more saturated
    black, white = np.zeros_like(GREY), np.full_like(GREY, 255.0)

    def kept(kind):
        return float((haze_tint(white, 1.0, kind) - haze_tint(black, 1.0, kind))[0, 0, 0]) / 255.0

    assert kept(STORM_KIND) == pytest.approx(0.25) and kept('dust') == pytest.approx(0.35)
    assert (r - b) > (dust[0] - dust[2])


def test_haze_tint_follows_the_image_channel_order():
    bgr = haze_tint(GREY, 1.0, STORM_KIND, bgr=True)[0, 0]
    rgb = haze_tint(GREY, 1.0, STORM_KIND)[0, 0]
    assert bgr == pytest.approx(rgb[::-1])


def test_degrade_rgb_keeps_shape_and_moves_the_particles():
    img = _scene()
    p = np.random.default_rng(1).random((PARTICLES, 4)).astype(np.float32)
    p0 = p.copy()
    out = degrade_rgb(img, 1.0, STORM_KIND, np.random.default_rng(2), p)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert not np.allclose(p, p0)                                                 # advanced in place
    mean = out.reshape(-1, 3).mean(axis=0)
    assert mean[0] > mean[1] > mean[2] and mean[0] - mean[2] > 80                  # the frame reads orange-red


def test_storm_drops_camera_confidence_well_below_half_and_below_plain_dust():
    clear = _scene()
    base = ImageQualityBaseline()
    rng = np.random.default_rng(3)
    for _ in range(6):                                                            # clear frames with sensor noise
        noisy = np.clip(clear + rng.normal(0, 1.5, clear.shape), 0, 255).astype(np.uint8)
        base.update(image_quality(noisy))
    assert base.confidence(image_quality(clear)) > 0.9

    def conf(level, kind):
        p = np.random.default_rng(1).random((PARTICLES, 4)).astype(np.float32)
        return base.confidence(image_quality(degrade_rgb(clear, level, kind, np.random.default_rng(7), p)))

    storm, dust = conf(1.0, STORM_KIND), conf(1.0, 'dust')
    assert storm < 0.35 and storm < dust                                          # CAUTION: camera conf < 0.5
    assert conf(0.3, STORM_KIND) > storm                                          # ramps with the level


# ---------------------------------------------------------------- anomaly command parsing
def test_anomaly_params_plain_defaults_are_unchanged():
    p = anomaly_params({'op': 'anomaly'})
    assert p == {'kind': 'dust', 'dust_level': 0.85, 'duration_s': 25.0, 'seismic_s': 4.0, 'wind_mps': 0.0}
    p = anomaly_params({'op': 'anomaly', 'dust_level': 1.7, 'duration_s': 10, 'wind_mps': 12.0})
    assert p['dust_level'] == 1.0 and p['duration_s'] == 10.0
    assert p['kind'] == 'dust' and p['wind_mps'] == 0.0                           # wind only comes with a storm


def test_anomaly_params_dust_storm():
    cmd = {'op': 'anomaly', 'kind': 'dust_storm', 'dust_level': 1.0, 'duration_s': 45, 'seismic_s': 0,
           'wind_mps': 14.0}
    assert anomaly_params(cmd) == {'kind': STORM_KIND, 'dust_level': 1.0, 'duration_s': 45.0, 'seismic_s': 0.0,
                                   'wind_mps': 14.0}
    assert anomaly_params({'op': 'anomaly', 'kind': 'dust_storm'}) == dict(STORM_DEFAULTS, kind=STORM_KIND)
    bad = anomaly_params({'op': 'anomaly', 'kind': 'dust_storm', 'dust_level': 'x', 'wind_mps': None,
                          'duration_s': float('nan'), 'seismic_s': -3})
    assert bad == dict(STORM_DEFAULTS, kind=STORM_KIND, seismic_s=0.0)            # bad values fall back, never raise
    assert anomaly_params({'kind': 'dust_storm', 'wind_mps': 500})['wind_mps'] <= 40.0


# ---------------------------------------------------------------- atmosphere state + /world/env_state
def _run(atm, t0, t1, dt=0.1):
    t = t0
    while t < t1 - 1e-9:
        t += dt
        atm.step(t, dt)
    return t


def test_plain_anomaly_env_state_and_echo():
    atm = Atmosphere()
    calm = atm.env_state(0.0)
    assert calm == {'dust_level': 0.0, 'seismic': False, 'remaining_s': None, 'kind': None, 'wind_mps': 0.0}
    ev = atm.command({'op': 'anomaly', 'dust_level': 0.85, 'duration_s': 25}, 0.0)
    assert ev == {'op': 'anomaly', 'ok': True, 'dust_level': 0.85, 'duration_s': 25.0, 'kind': 'dust',
                  'wind_mps': 0.0}
    t = _run(atm, 0.0, 3.0)
    env = atm.env_state(t)
    assert env['dust_level'] == pytest.approx(0.85) and env['kind'] == 'dust' and env['wind_mps'] == 0.0
    assert env['seismic'] is True and env['remaining_s'] == pytest.approx(22.0, abs=0.11)
    _run(atm, t, 30.0)
    assert atm.env_state(30.0) == calm


def test_storm_ramps_dust_and_wind_together_then_clears():
    atm = Atmosphere()
    ev = atm.command({'op': 'anomaly', 'kind': 'dust_storm', 'dust_level': 1.0, 'duration_s': 45, 'seismic_s': 0,
                      'wind_mps': 14.0}, 10.0)
    assert ev['kind'] == STORM_KIND and ev['wind_mps'] == 14.0 and ev['dust_level'] == 1.0
    t = _run(atm, 10.0, 10.0 + RAMP_S / 2)
    env = atm.env_state(t)
    assert env['kind'] == STORM_KIND and env['seismic'] is False
    assert env['dust_level'] == pytest.approx(0.5, abs=0.07)
    assert env['wind_mps'] == pytest.approx(14.0 * env['dust_level'], abs=0.05)   # ramped like the dust level
    t = _run(atm, t, 20.0)
    assert atm.env_state(t)['dust_level'] == 1.0 and atm.env_state(t)['wind_mps'] == 14.0
    assert atm.level == 1.0 and atm.kind == STORM_KIND
    t = _run(atm, t, 55.0 + RAMP_S / 2)                                          # the storm ends at 55 s
    env = atm.env_state(t)
    assert env['kind'] == STORM_KIND and 0.0 < env['dust_level'] < 1.0           # still the storm while it fades
    assert env['wind_mps'] == pytest.approx(14.0 * env['dust_level'], abs=0.05)
    t = _run(atm, t, 58.0)
    assert atm.env_state(t) == {'dust_level': 0.0, 'seismic': False, 'remaining_s': None, 'kind': None,
                                'wind_mps': 0.0}


def test_clear_anomaly_ends_a_storm_and_a_plain_anomaly_replaces_its_wind():
    atm = Atmosphere()
    atm.command({'op': 'anomaly', 'kind': 'dust_storm', 'wind_mps': 14.0}, 0.0)
    t = _run(atm, 0.0, 3.0)
    atm.clear(t)
    t = _run(atm, t, t + RAMP_S + 0.2)
    assert atm.env_state(t)['kind'] is None and atm.env_state(t)['wind_mps'] == 0.0
    atm.command({'op': 'anomaly', 'kind': 'dust_storm', 'wind_mps': 10.0}, t)
    t = _run(atm, t, t + 3.0)
    atm.command({'op': 'anomaly', 'dust_level': 0.5, 'duration_s': 20}, t)       # plain dust: the wind dies down
    t2 = _run(atm, t, t + RAMP_S / 2)
    assert 0.0 < atm.env_state(t2)['wind_mps'] < 10.0 and atm.env_state(t2)['kind'] == 'dust'
    t2 = _run(atm, t2, t + RAMP_S + 0.2)
    assert atm.env_state(t2)['wind_mps'] == 0.0 and atm.env_state(t2)['dust_level'] == pytest.approx(0.5)


def test_sim_time_going_backwards_never_ramps():
    atm = Atmosphere()
    atm.command({'op': 'anomaly', 'kind': 'dust_storm'}, 5.0)
    atm.step(4.0, -1.0)
    assert atm.level == 0.0 and atm.wind == 0.0
    assert math.isfinite(atm.env_state(4.0)['remaining_s'])
