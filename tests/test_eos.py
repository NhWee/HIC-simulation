import csv
import math
import warnings

import pytest
import torch

from hic import eos as eos_module
from hic.eos import (
    EquationOfState,
    consistent_temperatures,
    read_eos_table,
    table_segment_slopes,
    thermodynamic_consistency,
)


def energies_between_nodes(nodes: tuple[float, ...]) -> torch.Tensor:
    """Probe points that avoid the kinks of the piecewise-linear p(e)."""
    edges = (0.0, *nodes, nodes[-1] * 6.0)
    points = []
    for low, high in zip(edges, edges[1:]):
        points.append(torch.linspace(low, high, 41, dtype=torch.float64)[2:-2])
    return torch.cat(points)


def test_shipped_table_is_thermodynamically_consistent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        energy, pressure, sound_speed2, temperature = read_eos_table(eos_module.Path("data/eos_table.csv"))
    report = thermodynamic_consistency(energy, pressure, sound_speed2, temperature)
    assert report.max_temperature_log_error < 1.0e-4
    assert report.max_sound_speed2_error == 0.0
    assert all(0.0 <= slope <= 1.0 for slope in report.slopes)


def test_shipped_table_keeps_switching_temperature(table_eos):
    switching = table_eos.temperature(torch.tensor([0.35], dtype=torch.float64))
    assert float(switching) == pytest.approx(0.145, abs=1.0e-6)


def test_consistent_temperatures_reproduce_shipped_column(table_eos):
    anchor = table_eos.table_energy.index(0.35)
    rebuilt = consistent_temperatures(table_eos.table_energy, table_eos.table_pressure, anchor, 0.145)
    assert rebuilt == pytest.approx(table_eos.table_temperature, rel=1.0e-5)


def test_table_pressure_vanishes_in_vacuum_and_extrapolates(table_eos):
    energy = torch.tensor([0.0, 1.0e-8, 0.025, 6.0, 9.0], dtype=torch.float64)
    pressure = table_eos.pressure(energy)
    first_slope = table_eos.table_pressure[0] / table_eos.table_energy[0]
    last_slope = table_segment_slopes(table_eos.table_energy, table_eos.table_pressure)[-1]
    assert float(pressure[0]) == 0.0
    assert float(pressure[1]) == pytest.approx(first_slope * 1.0e-8)
    assert float(pressure[2]) == pytest.approx(first_slope * 0.025)
    assert float(pressure[3]) == pytest.approx(table_eos.table_pressure[-1])
    assert float(pressure[4]) == pytest.approx(table_eos.table_pressure[-1] + 3.0 * last_slope)


@pytest.mark.parametrize("name", ["conformal_eos", "crossover_eos", "table_eos"])
def test_sound_speed_is_pressure_derivative(name, request):
    model = request.getfixturevalue(name)
    nodes = model.table_energy or (0.5, 1.0, 2.0)
    energy = energies_between_nodes(nodes)
    step = 1.0e-7
    derivative = (model.pressure(energy + step) - model.pressure(energy - step)) / (2.0 * step)
    assert torch.allclose(model.sound_speed2(energy), derivative, rtol=1.0e-5, atol=1.0e-7)


@pytest.mark.parametrize("name", ["conformal_eos", "crossover_eos", "table_eos"])
def test_max_sound_speed_bounds_sound_speed(name, request):
    model = request.getfixturevalue(name)
    energy = torch.linspace(0.0, 40.0, 400_001, dtype=torch.float64)
    assert float(torch.max(model.sound_speed2(energy))) <= model.max_sound_speed() ** 2 + 1.0e-12


def test_crossover_sound_speed_exceeds_conformal_near_transition(crossover_eos):
    # p = cs2_eff(e) e with increasing cs2_eff gives dp/de > p/e; the Rusanov
    # signal speed must use the true maximum, not 1/sqrt(3).
    assert crossover_eos.max_sound_speed() ** 2 > 0.45


def test_table_temperature_obeys_thermodynamic_identity(table_eos):
    energy = energies_between_nodes(table_eos.table_energy)
    step = 1.0e-7
    log_derivative = (
        torch.log(table_eos.temperature(energy + step)) - torch.log(table_eos.temperature(energy - step))
    ) / (2.0 * step)
    expected = table_eos.sound_speed2(energy) / (energy + table_eos.pressure(energy))
    assert torch.allclose(log_derivative, expected, rtol=1.0e-4)


def test_table_temperature_is_continuous_at_nodes(table_eos):
    nodes = torch.tensor(table_eos.table_energy, dtype=torch.float64)
    below = table_eos.temperature(nodes * (1.0 - 1.0e-9))
    above = table_eos.temperature(nodes * (1.0 + 1.0e-9))
    assert torch.allclose(below, above, rtol=1.0e-7)
    assert torch.allclose(table_eos.temperature(nodes), torch.tensor(table_eos.table_temperature, dtype=torch.float64))


def write_table(path, rows, header=("energy_density", "pressure", "sound_speed2", "temperature")):
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        writer.writerows(rows)


def test_inconsistent_table_warns(tmp_path):
    path = tmp_path / "old_table.csv"
    write_table(path, [(0.35, 0.058, 0.17, 0.145), (0.60, 0.110, 0.19, 0.165), (1.00, 0.210, 0.22, 0.190)])
    with pytest.warns(UserWarning, match="not thermodynamically consistent"):
        read_eos_table(path)


def test_acausal_table_is_rejected(tmp_path):
    path = tmp_path / "acausal.csv"
    write_table(path, [(1.0, 0.2, 0.3, 0.2), (2.0, 1.5, 0.3, 0.25)])
    with pytest.raises(ValueError, match="causality"):
        read_eos_table(path)


def test_table_without_optional_columns(tmp_path):
    path = tmp_path / "pressure_only.csv"
    write_table(path, [(0.5, 0.1), (1.0, 0.25)], header=("energy_density", "pressure"))
    energy, pressure, sound_speed2, temperature = read_eos_table(path)
    assert temperature == ()
    assert sound_speed2 == pytest.approx((0.25, 0.3))
    model = EquationOfState("table", 1.0 / 3.0, 0.2, 0.3, 1.0, 0.25, energy, pressure, sound_speed2, temperature)
    assert float(model.temperature(torch.tensor([0.7]), fallback_temperature=0.15)) == pytest.approx(0.15)


def test_cli_writes_a_consistent_table(tmp_path, capsys):
    path = tmp_path / "old_table.csv"
    write_table(
        path,
        [(0.05, 0.006, 0.12, 0.090), (0.35, 0.058, 0.17, 0.145), (1.00, 0.210, 0.22, 0.190), (6.00, 1.900, 0.33, 0.370)],
    )
    output = tmp_path / "consistent.csv"
    eos_module.main(["--eos-table", str(path), "--write-consistent", str(output), "--anchor-energy", "0.35"])
    assert "NOT consistent" in capsys.readouterr().out
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        energy, _, _, temperature = read_eos_table(output)
    assert temperature[energy.index(0.35)] == pytest.approx(0.145)
    assert temperature[-1] < 0.370
    assert math.isfinite(sum(temperature))
