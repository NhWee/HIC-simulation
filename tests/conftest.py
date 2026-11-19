import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def run_from_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tools resolve defaults such as data/eos_table.csv against the working directory."""
    monkeypatch.chdir(ROOT)


@pytest.fixture
def cpu():
    import torch

    return torch.device("cpu")


@pytest.fixture
def table_eos():
    from hic.eos import EquationOfState, read_eos_table

    energy, pressure, sound_speed2, temperature = read_eos_table(ROOT / "data" / "eos_table.csv")
    return EquationOfState(
        kind="table",
        speed_of_sound2=1.0 / 3.0,
        low_speed_of_sound2=min(sound_speed2),
        high_speed_of_sound2=max(sound_speed2),
        transition_energy=1.0,
        transition_width=0.25,
        table_energy=energy,
        table_pressure=pressure,
        table_sound_speed2=sound_speed2,
        table_temperature=temperature,
    )


@pytest.fixture
def conformal_eos():
    from hic.eos import EquationOfState

    return EquationOfState("conformal", 1.0 / 3.0, 0.15, 1.0 / 3.0, 1.0, 0.25)


@pytest.fixture
def crossover_eos():
    from hic.eos import EquationOfState

    return EquationOfState("crossover", 1.0 / 3.0, 0.15, 1.0 / 3.0, 1.0, 0.25)
