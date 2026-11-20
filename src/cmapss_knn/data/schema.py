"""Column layout of the raw C-MAPSS files.

Each row of a raw file is one operating cycle (one flight) of one engine:
26 whitespace-separated numbers = unit id, cycle, 3 operational settings and
21 sensor channels. (The dataset readme loosely says "26 sensor measurements";
the 26 is the total column count.)
"""

from __future__ import annotations

SUBSET_NAMES: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

ID_COLUMN = "unit"
CYCLE_COLUMN = "cycle"
OP_SETTING_COLUMNS: tuple[str, ...] = ("op1", "op2", "op3")
SENSOR_COLUMNS: tuple[str, ...] = tuple(f"s{i}" for i in range(1, 22))
ALL_COLUMNS: tuple[str, ...] = (ID_COLUMN, CYCLE_COLUMN, *OP_SETTING_COLUMNS, *SENSOR_COLUMNS)
N_COLUMNS = len(ALL_COLUMNS)  # 26

RUL_COLUMN = "rul"
RUL_CLIPPED_COLUMN = "rul_clipped"


def train_filename(subset: str) -> str:
    return f"train_{subset}.txt"


def test_filename(subset: str) -> str:
    return f"test_{subset}.txt"


def rul_filename(subset: str) -> str:
    return f"RUL_{subset}.txt"
