"""Physical layout of the simulated conveyor and the knobs that drive the simulation.

Everything the rest of the simulator needs to know about "the belt" lives here,
so the assumptions are in one place and easy to point at from the README.
"""
from dataclasses import dataclass

BELT_LENGTH_M = 5000          # one-way length; the belt itself is a 10 km loop (carry + return)
BELT_SPEED_MS = 4.0           # nominal belt speed
STATION_SPACING_M = 250
N_STATIONS = BELT_LENGTH_M // STATION_SPACING_M   # 20

SAMPLE_INTERVAL_S = 60        # one reading per station per minute
RUN_HOURS = (6, 24)           # belt runs 06:00-24:00 (18 h/day), stopped 00:00-06:00
MAX_LOAD_TPH = 6000

# Stations with drive motors (head drive + mid-belt booster drive)
DRIVE_STATIONS = {"ST-01", "ST-11"}
# Terrain: inclines and curves wear faster and have higher baseline vibration / misalignment
INCLINE_STATIONS = {"ST-04", "ST-05", "ST-06", "ST-14", "ST-15"}
CURVE_STATIONS = {"ST-09", "ST-17"}


@dataclass(frozen=True)
class Station:
    id: str
    position_m: float
    terrain: str          # "flat" | "incline" | "curve"
    has_drive: bool


def build_stations() -> list[Station]:
    stations = []
    for i in range(N_STATIONS):
        sid = f"ST-{i + 1:02d}"
        terrain = "incline" if sid in INCLINE_STATIONS else "curve" if sid in CURVE_STATIONS else "flat"
        stations.append(Station(sid, i * STATION_SPACING_M + STATION_SPACING_M / 2, terrain, sid in DRIVE_STATIONS))
    return stations


STATIONS = build_stations()

# Signals every station reports. motor_current_a is only meaningful at drive stations.
SIGNALS = [
    "bearing_temp_c",
    "vibration_rms_mms",
    "belt_offset_mm",
    "belt_tension_kn",
    "belt_speed_ms",
    "load_tph",
    "ambient_temp_c",
    "dust_level",
    "motor_current_a",
]
