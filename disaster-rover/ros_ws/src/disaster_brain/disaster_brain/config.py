"""Brain constants. Mirror of web/brain/config.ts: keep in sync."""
import math

G = 100
RES = 2.0
HALF = 100.0
N = G * G

CFG = dict(
    confTau=40.0, horizon=30.0, mapPeriod=0.5,
    maxV=3.0, maxW=1.6, ambient=22.0, lidarMax=30.0, thermalRange=45.0, cameraRange=28.0,
    reflexWater=0.2, reflexTemp=52.0, reflexTilt=26.0, reflexWind=26.0, reflexFront=1.3,
    lowConfReturnFrac=0.12, staleFrame=0.6,
    hotTemp=120.0, shakeThresh=1.8, tornadoWind=13.0, tornadoPressure=35.0,
    hysteresis=0.08, minDwell=2.0,
    probeSpacing=26.0, probeRuleSpacing=20.0, probeRadius=22.0,
    batteryLow=25.0, batteryAgentEvent=30.0,
    batteryPerM=0.075, batteryPerS=0.1, batteryReserve=12.0,
)

WATER, HEAT, RUBBLE, WIND, SLOPE = 0, 1, 2, 3, 4
CH_NAMES = ['water', 'heat', 'rubble', 'wind', 'slope']
NCH = 5
BEHAVIOURS = ['explore', 'flee', 'high_ground', 'hold_still', 'place_probe', 'return_to_safety']
NO_DATA = -9999.0


def clamp(v, a, b):
    return a if v < a else b if v > b else v


def clamp01(v):
    return clamp(v, 0.0, 1.0)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


_DIRS = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']


def compass(a):
    return _DIRS[int(round(a / (math.pi / 4))) % 8]
