from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


FRAME_START = 1
FRAME_CONTACT = 48
FRAME_PREEQUILIBRIUM = 52
FRAME_HYDRO_START = 64
FRAME_HYDRO_END = 240
FRAME_END = 312
FPS = 24
TRANSVERSE_SCALE = 0.75
LONGITUDINAL_SCALE = 0.48
FIELD_THRESHOLD_GEV = 0.145
QGP_CORE_TEMPERATURE_GEV = 0.165


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a data-driven heavy-ion collision scene in Blender.")
    parser.add_argument("--cache", type=Path, default=Path("outputs/blender_physics_event.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hic_blender_physics_scene.blend"))
    parser.add_argument("--render-animation", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--field-stride", type=int, default=2)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(arguments)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def new_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_object(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    collection.objects.link(obj)


def material_input(shader: bpy.types.Node, name: str, fallback_index: int | None = None):
    socket = shader.inputs.get(name)
    if socket is None and fallback_index is not None and fallback_index < len(shader.inputs):
        socket = shader.inputs[fallback_index]
    return socket


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.42,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    material_input(shader, "Base Color", 0).default_value = color
    material_input(shader, "Metallic", 1).default_value = metallic
    material_input(shader, "Roughness", 2).default_value = roughness
    emission = material_input(shader, "Emission Color") or material_input(shader, "Emission")
    if emission is not None:
        emission.default_value = color
    emission_socket = material_input(shader, "Emission Strength")
    if emission_socket is not None:
        emission_socket.default_value = emission_strength
    return material


def make_temperature_material() -> bpy.types.Material:
    material = bpy.data.materials.new("hydrodynamic_temperature_field")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = "temperature_color"
    links.new(vertex_color.outputs["Color"], material_input(shader, "Base Color", 0))
    emission = material_input(shader, "Emission Color") or material_input(shader, "Emission")
    if emission is not None:
        links.new(vertex_color.outputs["Color"], emission)
    emission_strength = material_input(shader, "Emission Strength")
    if emission_strength is not None:
        emission_strength.default_value = 0.38
    material_input(shader, "Metallic", 1).default_value = 0.0
    material_input(shader, "Roughness", 2).default_value = 0.48
    return material


def temperature_color(temperature: float) -> tuple[float, float, float, float]:
    stops = (
        (0.145, (0.78, 0.045, 0.008)),
        (0.165, (1.00, 0.20, 0.012)),
        (0.230, (1.00, 0.66, 0.055)),
        (0.370, (1.00, 0.96, 0.70)),
    )
    if temperature <= stops[0][0]:
        return (*stops[0][1], 1.0)
    for (left_value, left_color), (right_value, right_color) in zip(stops, stops[1:]):
        if temperature <= right_value:
            fraction = (temperature - left_value) / max(right_value - left_value, 1.0e-12)
            color = tuple(left + fraction * (right - left) for left, right in zip(left_color, right_color))
            return (*color, 1.0)
    return (*stops[-1][1], 1.0)


def preequilibrium_color(
    effective_temperature: float,
    matching_progress: float,
) -> tuple[float, float, float, float]:
    fraction = np.clip(
        (effective_temperature - FIELD_THRESHOLD_GEV) / (0.650 - FIELD_THRESHOLD_GEV),
        0.0,
        1.0,
    )
    stops = (
        (0.0, (0.015, 0.055, 0.22)),
        (0.35, (0.015, 0.24, 0.88)),
        (0.70, (0.02, 0.78, 1.00)),
        (1.0, (0.95, 0.90, 0.55)),
    )
    for (left_value, left_color), (right_value, right_color) in zip(stops, stops[1:]):
        if fraction <= right_value:
            interpolation = (fraction - left_value) / max(right_value - left_value, 1.0e-12)
            color = tuple(left + interpolation * (right - left) for left, right in zip(left_color, right_color))
            blue_color = color
            break
    else:
        blue_color = stops[-1][1]
    thermal_color = temperature_color(effective_temperature)[:3]
    blend = np.clip((matching_progress - 0.70) / 0.30, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    color = tuple(blue + blend * (thermal - blue) for blue, thermal in zip(blue_color, thermal_color))
    return (*color, 1.0)


def set_constant_interpolation(obj: bpy.types.Object, data_path: str) -> None:
    # Boolean visibility channels are evaluated discretely. Location keys use
    # the global LINEAR insertion preference configured for this scene.
    return


def set_linear_location(obj: bpy.types.Object) -> None:
    return


def key_visibility(obj: bpy.types.Object, frame: int, visible: bool) -> None:
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)
    set_constant_interpolation(obj, "hide_viewport")
    set_constant_interpolation(obj, "hide_render")


def show_between(obj: bpy.types.Object, first: int, last: int) -> None:
    if first > FRAME_START:
        key_visibility(obj, FRAME_START, False)
        key_visibility(obj, first - 1, False)
    key_visibility(obj, first, True)
    key_visibility(obj, last, True)
    if last < FRAME_END:
        key_visibility(obj, last + 1, False)


def key_location(obj: bpy.types.Object, frame: int, location: tuple[float, float, float]) -> None:
    obj.location = location
    obj.keyframe_insert(data_path="location", frame=frame)


def make_icosphere_mesh(name: str, subdivisions: int = 2) -> bpy.types.Mesh:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions, radius=1.0)
    temporary = bpy.context.object
    mesh = temporary.data
    mesh.name = name
    bpy.data.objects.remove(temporary, do_unlink=True)
    return mesh


def add_shared_sphere(
    name: str,
    mesh: bpy.types.Mesh,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    radius: float,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, mesh)
    link_object(obj, collection)
    obj.scale = (radius, radius, radius)
    if len(obj.data.materials) == 0:
        obj.data.materials.append(material)
    else:
        # Shared meshes are grouped by material by using a material-specific mesh copy.
        obj.material_slots[0].material = material
    return obj


def material_mesh_copy(base_mesh: bpy.types.Mesh, name: str, material: bpy.types.Material) -> bpy.types.Mesh:
    mesh = base_mesh.copy()
    mesh.name = name
    mesh.materials.clear()
    mesh.materials.append(material)
    return mesh


def add_nuclei(
    cache: np.lib.npyio.NpzFile,
    materials: dict[str, bpy.types.Material],
    collection: bpy.types.Collection,
) -> None:
    base_mesh = make_icosphere_mesh("nucleon_geometry", subdivisions=2)
    proton_mesh = material_mesh_copy(base_mesh, "proton_geometry", materials["nuclear_proton"])
    neutron_mesh = material_mesh_copy(base_mesh, "neutron_geometry", materials["neutron"])
    bpy.data.meshes.remove(base_mesh)
    impact = float(cache["impact_parameter"])

    for nucleus_name, positions_key, participant_key, direction, impact_center in (
        ("A", "nucleus_a_xyz", "participant_a", 1.0, -0.5 * impact),
        ("B", "nucleus_b_xyz", "participant_b", -1.0, 0.5 * impact),
    ):
        positions = cache[positions_key]
        participant = cache[participant_key].astype(bool)
        for index, local in enumerate(positions):
            is_proton = index < 79
            mesh = proton_mesh if is_proton else neutron_mesh
            obj = bpy.data.objects.new(f"nucleus_{nucleus_name}_{index:03d}", mesh)
            link_object(obj, collection)
            obj.scale = (0.20, 0.20, 0.20)
            local_beam = float(local[2]) * 0.035
            transverse_y = TRANSVERSE_SCALE * (float(local[1]) + impact_center)
            transverse_z = TRANSVERSE_SCALE * float(local[0])
            start_x = -direction * 15.5 + local_beam
            contact_x = local_beam
            exit_x = direction * 17.0 + local_beam
            key_location(obj, FRAME_START, (start_x, transverse_y, transverse_z))
            key_location(obj, FRAME_CONTACT, (contact_x, transverse_y, transverse_z))
            if participant[index]:
                key_visibility(obj, FRAME_START, True)
                key_visibility(obj, FRAME_CONTACT + 2, True)
                key_visibility(obj, FRAME_CONTACT + 3, False)
            else:
                key_location(obj, FRAME_CONTACT + 2, (contact_x + 0.2 * direction, transverse_y, transverse_z))
                key_location(obj, FRAME_HYDRO_START + 22, (exit_x, transverse_y, transverse_z))
                key_visibility(obj, FRAME_START, True)
                key_visibility(obj, FRAME_HYDRO_START + 22, True)
                key_visibility(obj, FRAME_HYDRO_START + 23, False)
            set_linear_location(obj)


def add_binary_collisions(
    cache: np.lib.npyio.NpzFile,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> None:
    points = cache["collision_centers"]
    if points.size == 0:
        return
    sample_count = min(180, points.shape[0])
    indices = np.linspace(0, points.shape[0] - 1, sample_count, dtype=int)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    temporary = bpy.context.object
    mesh = temporary.data
    mesh.name = "binary_collision_geometry"
    mesh.materials.append(material)
    bpy.data.objects.remove(temporary, do_unlink=True)
    for display_index, collision_index in enumerate(indices):
        point = points[collision_index]
        obj = bpy.data.objects.new(f"binary_collision_{display_index:03d}", mesh)
        link_object(obj, collection)
        obj.location = (0.0, TRANSVERSE_SCALE * float(point[1]), TRANSVERSE_SCALE * float(point[0]))
        obj.scale = (0.075, 0.075, 0.075)
        show_between(obj, FRAME_CONTACT, FRAME_PREEQUILIBRIUM + 1)


def preequilibrium_frame_boundaries(preeq_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau_start = float(preeq_tau[0])
    tau_end = float(preeq_tau[-1])
    denominator = max(tau_end - tau_start, 1.0e-12)
    mapped = np.rint(
        FRAME_PREEQUILIBRIUM
        + (preeq_tau - tau_start) / denominator * (FRAME_HYDRO_START - 1 - FRAME_PREEQUILIBRIUM)
    ).astype(int)
    first = np.empty_like(mapped)
    last = np.empty_like(mapped)
    for index in range(mapped.size):
        first[index] = FRAME_PREEQUILIBRIUM if index == 0 else (mapped[index - 1] + mapped[index]) // 2 + 1
        last[index] = FRAME_HYDRO_START - 1 if index == mapped.size - 1 else (mapped[index] + mapped[index + 1]) // 2
    return mapped, first, last


def hydro_frame_boundaries(hydro_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau0 = float(hydro_tau[0])
    tau_end = float(hydro_tau[-1])
    denominator = max(tau_end - tau0, 1.0e-12)
    mapped = np.rint(
        FRAME_HYDRO_START + (hydro_tau - tau0) / denominator * (FRAME_HYDRO_END - FRAME_HYDRO_START)
    ).astype(int)
    first = np.empty_like(mapped)
    last = np.empty_like(mapped)
    for index in range(mapped.size):
        first[index] = FRAME_HYDRO_START if index == 0 else (mapped[index - 1] + mapped[index]) // 2 + 1
        last[index] = FRAME_HYDRO_END if index == mapped.size - 1 else (mapped[index] + mapped[index + 1]) // 2
    return mapped, first, last


def build_temperature_mesh(
    name: str,
    temperature: np.ndarray,
    axis: np.ndarray,
    tau: float,
    eta_window: float,
    stride: int,
    color_mapper=temperature_color,
) -> bpy.types.Mesh:
    sampled_temperature = temperature[::stride, ::stride].astype(np.float64)
    sampled_axis = axis[::stride].astype(np.float64)
    rows, columns = sampled_temperature.shape
    threshold_ratio = np.maximum(sampled_temperature / FIELD_THRESHOLD_GEV, 1.0)
    local_eta = 0.70 * np.sqrt(2.0 * np.log(threshold_ratio))
    local_eta = np.minimum(local_eta, eta_window)
    half_length = LONGITUDINAL_SCALE * tau * np.sinh(local_eta)

    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    for side in (1.0, -1.0):
        for row in range(rows):
            for column in range(columns):
                vertices.append(
                    (
                        side * float(half_length[row, column]),
                        TRANSVERSE_SCALE * float(sampled_axis[row]),
                        TRANSVERSE_SCALE * float(sampled_axis[column]),
                    )
                )
                colors.append(color_mapper(float(sampled_temperature[row, column])))

    layer_size = rows * columns
    active = sampled_temperature >= FIELD_THRESHOLD_GEV
    faces: list[tuple[int, int, int, int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            if not bool(np.all(active[row : row + 2, column : column + 2])):
                continue
            a = row * columns + column
            b = a + 1
            d = (row + 1) * columns + column
            c = d + 1
            faces.append((a, b, c, d))
            faces.append((layer_size + d, layer_size + c, layer_size + b, layer_size + a))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    color_layer = mesh.color_attributes.new(name="temperature_color", type="FLOAT_COLOR", domain="POINT")
    for color_value, color in zip(color_layer.data, colors):
        color_value.color = color
    return mesh


def marching_squares_segments(
    temperature: np.ndarray,
    axis: np.ndarray,
    threshold: float,
    stride: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    values = temperature[::stride, ::stride].astype(np.float64)
    coordinates = axis[::stride].astype(np.float64)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def crossing(
        x0: float,
        y0: float,
        value0: float,
        x1: float,
        y1: float,
        value1: float,
    ) -> tuple[float, float] | None:
        if (value0 >= threshold) == (value1 >= threshold):
            return None
        denominator = value1 - value0
        if abs(denominator) < 1.0e-12:
            return None
        fraction = (threshold - value0) / denominator
        return x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)

    for row in range(values.shape[0] - 1):
        for column in range(values.shape[1] - 1):
            x0 = float(coordinates[column])
            x1 = float(coordinates[column + 1])
            y0 = float(coordinates[row])
            y1 = float(coordinates[row + 1])
            v00 = float(values[row, column])
            v10 = float(values[row, column + 1])
            v11 = float(values[row + 1, column + 1])
            v01 = float(values[row + 1, column])
            intersections = [
                crossing(x0, y0, v00, x1, y0, v10),
                crossing(x1, y0, v10, x1, y1, v11),
                crossing(x1, y1, v11, x0, y1, v01),
                crossing(x0, y1, v01, x0, y0, v00),
            ]
            points = [point for point in intersections if point is not None]
            if len(points) == 2:
                segments.append((points[0], points[1]))
            elif len(points) == 4:
                segments.append((points[0], points[1]))
                segments.append((points[2], points[3]))
    return segments


def add_contour_curve(
    name: str,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object | None:
    if not segments:
        return None
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = 0.035
    curve.bevel_resolution = 2
    curve.materials.append(material)
    for start, end in segments:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (0.0, TRANSVERSE_SCALE * start[1], TRANSVERSE_SCALE * start[0], 1.0)
        spline.points[1].co = (0.0, TRANSVERSE_SCALE * end[1], TRANSVERSE_SCALE * end[0], 1.0)
    obj = bpy.data.objects.new(name, curve)
    link_object(obj, collection)
    return obj


def add_hydrodynamic_field(
    cache: np.lib.npyio.NpzFile,
    temperature_material: bpy.types.Material,
    freezeout_material: bpy.types.Material,
    field_collection: bpy.types.Collection,
    surface_collection: bpy.types.Collection,
    qgp_collection: bpy.types.Collection,
    qgp_material: bpy.types.Material,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = cache["grid_axis"]
    hydro_tau = cache["hydro_tau"]
    temperature_frames = cache["temperature"]
    eta_window = float(cache["eta_window"])
    mapped, first, last = hydro_frame_boundaries(hydro_tau)
    freezeout_temperature = float(cache["freezeout_temperature"])

    for index, (tau, temperature) in enumerate(zip(hydro_tau, temperature_frames)):
        mesh = build_temperature_mesh(
            f"temperature_mesh_{index:03d}",
            temperature,
            axis,
            float(tau),
            eta_window,
            stride,
        )
        mesh.materials.append(temperature_material)
        field = bpy.data.objects.new(f"hydro_field_{index:03d}", mesh)
        link_object(field, field_collection)
        field["tau_fm_c"] = float(tau)
        field["max_temperature_gev"] = float(np.max(temperature))
        show_between(field, int(first[index]), int(last[index]))

        segments = marching_squares_segments(temperature, axis, freezeout_temperature, stride)
        contour = add_contour_curve(
            f"freezeout_contour_{index:03d}",
            segments,
            freezeout_material,
            surface_collection,
        )
        if contour is not None:
            contour["tau_fm_c"] = float(tau)
            contour["temperature_gev"] = freezeout_temperature
            show_between(contour, int(first[index]), int(last[index]))

        qgp_segments = marching_squares_segments(temperature, axis, QGP_CORE_TEMPERATURE_GEV, stride)
        qgp_contour = add_contour_curve(
            f"qgp_core_contour_{index:03d}",
            qgp_segments,
            qgp_material,
            qgp_collection,
        )
        if qgp_contour is not None:
            qgp_contour["tau_fm_c"] = float(tau)
            qgp_contour["temperature_gev"] = QGP_CORE_TEMPERATURE_GEV
            show_between(qgp_contour, int(first[index]), int(last[index]))
    return mapped, first, last


def add_preequilibrium_field(
    cache: np.lib.npyio.NpzFile,
    field_material: bpy.types.Material,
    collection: bpy.types.Collection,
    stride: int,
) -> None:
    axis = cache["grid_axis"]
    preeq_tau = cache["preeq_tau"]
    temperature_frames = cache["preeq_effective_temperature"]
    eta_window = float(cache["eta_window"])
    _, first, last = preequilibrium_frame_boundaries(preeq_tau)

    for index, (tau, effective_temperature) in enumerate(zip(preeq_tau, temperature_frames)):
        matching_progress = index / max(len(preeq_tau) - 1, 1)
        mesh = build_temperature_mesh(
            f"preequilibrium_mesh_{index:03d}",
            effective_temperature,
            axis,
            float(tau),
            eta_window,
            stride,
            color_mapper=lambda value, progress=matching_progress: preequilibrium_color(value, progress),
        )
        mesh.materials.append(field_material)
        field = bpy.data.objects.new(f"preequilibrium_field_{index:03d}", mesh)
        link_object(field, collection)
        field["tau_fm_c"] = float(tau)
        field["max_t00_gev_fm3"] = float(np.max(cache["preeq_energy_density"][index]))
        field["max_landau_energy_gev_fm3"] = float(np.max(cache["preeq_landau_energy_density"][index]))
        field["max_effective_temperature_gev"] = float(np.max(effective_temperature))
        field["quantity"] = "Landau-frame effective temperature; non-equilibrium diagnostic"
        show_between(field, int(first[index]), int(last[index]))


def tau_to_frame(tau: float, hydro_tau: np.ndarray) -> int:
    tau0 = float(hydro_tau[0])
    tau_end = float(hydro_tau[-1])
    fraction = (tau - tau0) / max(tau_end - tau0, 1.0e-12)
    return int(round(FRAME_HYDRO_START + np.clip(fraction, 0.0, 1.0) * (FRAME_HYDRO_END - FRAME_HYDRO_START)))


def add_particles(
    cache: np.lib.npyio.NpzFile,
    materials: dict[str, bpy.types.Material],
    collection: bpy.types.Collection,
) -> np.ndarray:
    base_mesh = make_icosphere_mesh("hadron_geometry", subdivisions=2)
    species_names = [str(value) for value in cache["particle_species_names"].tolist()]
    meshes = {
        index: material_mesh_copy(base_mesh, f"{name}_geometry", materials[name])
        for index, name in enumerate(species_names)
    }
    bpy.data.meshes.remove(base_mesh)
    species = cache["particle_species"]
    tau_values = cache["particle_tau"]
    eta_values = cache["particle_eta_s"]
    x_values = cache["particle_x"]
    y_values = cache["particle_y"]
    energy = cache["particle_energy"]
    momentum_longitudinal = cache["particle_momentum_longitudinal"]
    momentum_x = cache["particle_momentum_x"]
    momentum_y = cache["particle_momentum_y"]
    hydro_tau = cache["hydro_tau"]
    final_time = float(hydro_tau[-1]) + 4.0
    release_frames = np.empty(species.size, dtype=np.int32)
    radii = (0.075, 0.095, 0.115)

    for index in range(species.size):
        species_index = int(species[index])
        obj = bpy.data.objects.new(f"hadron_{index:04d}_{species_names[species_index]}", meshes[species_index])
        link_object(obj, collection)
        radius = radii[species_index]
        obj.scale = (radius, radius, radius)
        tau = float(tau_values[index])
        eta_s = float(eta_values[index])
        release_frame = tau_to_frame(tau, hydro_tau)
        release_frames[index] = release_frame
        start = (
            LONGITUDINAL_SCALE * tau * math.sinh(eta_s),
            TRANSVERSE_SCALE * float(y_values[index]),
            TRANSVERSE_SCALE * float(x_values[index]),
        )
        propagation_time = max(final_time - tau, 0.0)
        inverse_energy = 1.0 / max(float(energy[index]), 1.0e-8)
        final = (
            start[0] + LONGITUDINAL_SCALE * float(momentum_longitudinal[index]) * inverse_energy * propagation_time,
            start[1] + TRANSVERSE_SCALE * float(momentum_y[index]) * inverse_energy * propagation_time,
            start[2] + TRANSVERSE_SCALE * float(momentum_x[index]) * inverse_energy * propagation_time,
        )
        key_location(obj, release_frame, start)
        key_location(obj, FRAME_END, final)
        key_visibility(obj, FRAME_START, False)
        if release_frame > FRAME_START:
            key_visibility(obj, release_frame - 1, False)
        key_visibility(obj, release_frame, True)
        key_visibility(obj, FRAME_END, True)
        set_linear_location(obj)
        obj["emission_tau_fm_c"] = tau
        obj["species"] = species_names[species_index]
        obj["free_streaming"] = True
        obj["ballistic_location_keys"] = 2
    return release_frames


def add_beam_axis(material: bpy.types.Material, collection: bpy.types.Collection) -> None:
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.018, depth=34.0, location=(0.0, 0.0, 0.0))
    axis = bpy.context.object
    axis.name = "beam_axis"
    axis.rotation_euler[1] = math.pi / 2.0
    axis.data.materials.append(material)
    for source_collection in list(axis.users_collection):
        source_collection.objects.unlink(axis)
    link_object(axis, collection)


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_camera_and_lights() -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=(7.0, -9.0, 13.0))
    key_light = bpy.context.object
    key_light.name = "key_light"
    key_light.data.energy = 1150
    key_light.data.shape = "DISK"
    key_light.data.size = 8.0
    look_at(key_light, (0.0, 0.0, 0.0))

    bpy.ops.object.light_add(type="AREA", location=(-10.0, 4.0, 5.0))
    fill_light = bpy.context.object
    fill_light.name = "fill_light"
    fill_light.data.energy = 700
    fill_light.data.color = (0.30, 0.55, 1.0)
    fill_light.data.size = 7.0
    look_at(fill_light, (0.0, 0.0, 0.0))

    bpy.ops.object.light_add(type="AREA", location=(10.0, 7.0, 2.0))
    rim_light = bpy.context.object
    rim_light.name = "rim_light"
    rim_light.data.energy = 650
    rim_light.data.color = (1.0, 0.33, 0.08)
    rim_light.data.size = 5.0
    look_at(rim_light, (0.0, 0.0, 0.0))

    bpy.ops.object.camera_add(location=(24.0, -27.0, 18.0))
    camera = bpy.context.object
    camera.name = "fixed_physics_camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 29.5
    look_at(camera, (0.0, 0.0, 0.0))
    bpy.context.scene.camera = camera
    return camera


def add_text_overlay(
    name: str,
    body: str,
    camera: bpy.types.Object,
    material: bpy.types.Material,
    location: tuple[float, float, float],
    size: float,
) -> bpy.types.Object:
    bpy.ops.object.text_add()
    text = bpy.context.object
    text.name = name
    text.data.body = body
    text.data.align_x = "LEFT"
    text.data.align_y = "CENTER"
    text.data.extrude = 0.006
    text.data.bevel_depth = 0.001
    text.data.materials.append(material)
    text.parent = camera
    text.location = location
    text.rotation_euler = (0.0, 0.0, 0.0)
    text.scale = (size, size, size)
    return text


def add_stage_labels(
    cache: np.lib.npyio.NpzFile,
    camera: bpy.types.Object,
    material: bpy.types.Material,
    release_frames: np.ndarray,
) -> None:
    raw_particlization_frame = int(np.quantile(release_frames, 0.35)) if release_frames.size else FRAME_HYDRO_END
    particlization_frame = max(FRAME_HYDRO_START + 20, min(raw_particlization_frame, FRAME_HYDRO_END))
    stages = (
        ("stage_incoming", "INCOMING Au NUCLEI", FRAME_START, FRAME_CONTACT - 1),
        ("stage_initialization", "MC-GLAUBER ENERGY DEPOSITION", FRAME_CONTACT, FRAME_PREEQUILIBRIUM - 1),
        (
            "stage_preequilibrium",
            "COLLISIONLESS PRE-EQUILIBRIUM",
            FRAME_PREEQUILIBRIUM,
            FRAME_HYDRO_START - 4,
        ),
        (
            "stage_landau_map",
            "LANDAU-FRAME ENERGY AND PREFLOW",
            FRAME_HYDRO_START - 3,
            FRAME_HYDRO_START - 1,
        ),
        (
            "stage_hydrodynamization",
            "LANDAU MATCH: QGP CORE  T >= 165 MeV",
            FRAME_HYDRO_START,
            FRAME_HYDRO_START + 11,
        ),
        (
            "stage_hydro",
            "2+1D RELATIVISTIC IDEAL HYDRODYNAMICS",
            FRAME_HYDRO_START + 12,
            particlization_frame - 1,
        ),
        ("stage_particlization", "COOPER-FRYE PARTICLIZATION", particlization_frame, FRAME_HYDRO_END),
        ("stage_free_streaming", "SAMPLED HADRONS: FREE STREAMING", FRAME_HYDRO_END + 1, FRAME_END),
    )
    for name, body, first, last in stages:
        label = add_text_overlay(name, body, camera, material, (-11.4, 6.25, -10.0), 0.30)
        show_between(label, first, last)

    impact = float(cache["impact_parameter"])
    n_part = int(cache["n_part"])
    n_coll = int(cache["n_coll"])
    metadata = add_text_overlay(
        "physics_metadata",
        f"Au+Au  sqrt(sNN)=200 GeV  b={impact:.1f} fm  Npart={n_part}  Ncoll={n_coll}",
        camera,
        material,
        (-11.4, -5.75, -10.0),
        0.17,
    )
    key_visibility(metadata, FRAME_START, True)
    model = add_text_overlay(
        "model_metadata",
        "FREE STREAM 0.1-0.6 fm/c  |  QGP CORE 165 MeV  |  Tsw=145 MeV  |  NO AFTERBURNER",
        camera,
        material,
        (-11.4, -6.25, -10.0),
        0.15,
    )
    key_visibility(model, FRAME_START, True)


def add_time_diagnostics(
    cache: np.lib.npyio.NpzFile,
    camera: bpy.types.Object,
    material: bpy.types.Material,
) -> None:
    preeq_tau = cache["preeq_tau"]
    preeq_energy = cache["preeq_landau_energy_density"]
    preeq_temperature = cache["preeq_effective_temperature"]
    preeq_speed = np.hypot(cache["preeq_velocity_x"], cache["preeq_velocity_y"])
    _, preeq_first, preeq_last = preequilibrium_frame_boundaries(preeq_tau)
    for index, tau in enumerate(preeq_tau):
        energy = np.maximum(preeq_energy[index], 0.0)
        mean_flow = float(np.sum(energy * preeq_speed[index]) / max(float(np.sum(energy)), 1.0e-12))
        label = add_text_overlay(
            f"preeq_diagnostic_{index:03d}",
            f"tau={float(tau):.3f} fm/c  |  max T_eff,L={1000.0 * float(np.max(preeq_temperature[index])):.0f} MeV"
            f"  |  <vT>_e={mean_flow:.3f} c",
            camera,
            material,
            (-11.4, 5.72, -10.0),
            0.17,
        )
        show_between(label, int(preeq_first[index]), int(preeq_last[index]))

    hydro_tau = cache["hydro_tau"]
    hydro_temperature = cache["temperature"]
    _, hydro_first, hydro_last = hydro_frame_boundaries(hydro_tau)
    for index, tau in enumerate(hydro_tau):
        qgp_cells = int(np.count_nonzero(hydro_temperature[index] >= QGP_CORE_TEMPERATURE_GEV))
        label = add_text_overlay(
            f"hydro_diagnostic_{index:03d}",
            f"tau={float(tau):.3f} fm/c  |  max T={1000.0 * float(np.max(hydro_temperature[index])):.0f} MeV"
            f"  |  QGP-core cells={qgp_cells}",
            camera,
            material,
            (-11.4, 5.72, -10.0),
            0.17,
        )
        show_between(label, int(hydro_first[index]), int(hydro_last[index]))


def configure_render(scene: bpy.types.Scene) -> None:
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.render.fps = FPS
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.006, 0.008, 0.012)
    scene.view_settings.look = "AgX - Medium High Contrast"


def add_scene_metadata(scene: bpy.types.Scene, cache_path: Path, cache: np.lib.npyio.NpzFile) -> None:
    scene["event_cache"] = str(cache_path.resolve())
    scene["format_version"] = str(cache["format_version"])
    scene["physics_model"] = str(cache["physics_model"])
    scene["afterburner"] = str(cache["afterburner"])
    scene["impact_parameter_fm"] = float(cache["impact_parameter"])
    scene["seed"] = int(cache["seed"])
    scene["n_part"] = int(cache["n_part"])
    scene["n_coll"] = int(cache["n_coll"])
    scene["preequilibrium_tau_start_fm_c"] = float(cache["preeq_tau"][0])
    scene["hydrodynamization_tau_fm_c"] = float(cache["hydro_tau"][0])
    scene["mean_pt_gev"] = float(cache["mean_pt"])
    scene["v2_reaction_plane"] = float(cache["v2"])


def audit_scene(scene: bpy.types.Scene, cache: np.lib.npyio.NpzFile) -> None:
    expected_particles = int(cache["particle_species"].size)
    particle_objects = [obj for obj in bpy.data.objects if obj.name.startswith("hadron_")]
    field_objects = [obj for obj in bpy.data.objects if obj.name.startswith("hydro_field_")]
    preequilibrium_objects = [obj for obj in bpy.data.objects if obj.name.startswith("preequilibrium_field_")]
    if len(particle_objects) != expected_particles:
        raise RuntimeError(f"Particle object mismatch: {len(particle_objects)} != {expected_particles}")
    if len(field_objects) != int(cache["hydro_tau"].size):
        raise RuntimeError("Hydrodynamic field snapshot count does not match the numerical cache.")
    if len(preequilibrium_objects) != int(cache["preeq_tau"].size):
        raise RuntimeError("Pre-equilibrium field snapshot count does not match the numerical cache.")
    if scene.camera is None or scene.camera.animation_data is not None:
        raise RuntimeError("The physics camera must be fixed and unanimated.")
    for particle in particle_objects:
        if int(particle.get("ballistic_location_keys", 0)) != 2:
            raise RuntimeError(f"Non-ballistic particle trajectory detected: {particle.name}")

    scene.frame_set(FRAME_PREEQUILIBRIUM)
    visible_preequilibrium = sum(not obj.hide_render for obj in preequilibrium_objects)
    scene.frame_set(FRAME_HYDRO_START)
    visible_fields = sum(not obj.hide_render for obj in field_objects)
    visible_preequilibrium_at_hydro = sum(not obj.hide_render for obj in preequilibrium_objects)
    scene.frame_set(FRAME_END)
    visible_final_particles = sum(not obj.hide_render for obj in particle_objects)
    visible_final_fields = sum(not obj.hide_render for obj in field_objects)
    if (
        visible_preequilibrium != 1
        or visible_fields != 1
        or visible_preequilibrium_at_hydro != 0
        or visible_final_particles != expected_particles
        or visible_final_fields != 0
    ):
        raise RuntimeError(
            "Scene visibility audit failed: "
            f"preeq_fields={visible_preequilibrium}, hydro_start_fields={visible_fields}, "
            f"preeq_at_hydro={visible_preequilibrium_at_hydro}, "
            f"final_particles={visible_final_particles}, final_fields={visible_final_fields}"
        )
    print(
        "Scene audit passed: "
        f"preeq={len(preequilibrium_objects)}, fields={len(field_objects)}, particles={len(particle_objects)}, "
        "fixed_camera=true, ballistic_tracks=true"
    )


def build_scene(cache_path: Path, field_stride: int) -> tuple[bpy.types.Scene, np.lib.npyio.NpzFile, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Physics event cache not found: {cache_path}")
    cache = np.load(cache_path, allow_pickle=False)
    if str(cache["format_version"]) != "hic_blender_event_v5":
        raise RuntimeError(f"Unsupported cache format: {cache['format_version']}")

    clear_scene()
    scene = bpy.context.scene
    configure_render(scene)

    nuclei_collection = new_collection("01_Nuclei")
    collision_collection = new_collection("02_Glauber_Collisions")
    preequilibrium_collection = new_collection("03_PreEquilibrium_Field")
    field_collection = new_collection("04_Hydrodynamic_Field")
    qgp_collection = new_collection("05_QGP_Core_Crossover")
    surface_collection = new_collection("06_Freezeout_Surface")
    particle_collection = new_collection("07_Cooper_Frye_Particles")
    reference_collection = new_collection("08_Reference")

    materials = {
        "nuclear_proton": make_material("nuclear_proton", (0.80, 0.08, 0.025, 1.0), roughness=0.34),
        "neutron": make_material("nuclear_neutron", (0.055, 0.24, 0.70, 1.0), roughness=0.34),
        "collision": make_material("binary_collision", (1.0, 0.78, 0.12, 1.0), emission_strength=4.0),
        "freezeout": make_material("freezeout_isotherm", (0.18, 0.92, 1.0, 1.0), emission_strength=2.5),
        "qgp_core": make_material("qgp_core_crossover", (1.0, 0.72, 0.06, 1.0), emission_strength=3.5),
        "pion": make_material("pion", (0.12, 0.64, 1.0, 1.0), emission_strength=0.35),
        "kaon": make_material("kaon", (0.20, 0.86, 0.34, 1.0), emission_strength=0.35),
        "proton": make_material("proton_hadron", (1.0, 0.24, 0.10, 1.0), emission_strength=0.35),
        "text": make_material("text", (0.95, 0.98, 1.0, 1.0), emission_strength=3.0),
        "reference": make_material("reference", (0.08, 0.10, 0.14, 1.0), roughness=0.65),
    }
    temperature_material = make_temperature_material()

    add_nuclei(cache, materials, nuclei_collection)
    add_binary_collisions(cache, materials["collision"], collision_collection)
    add_preequilibrium_field(cache, temperature_material, preequilibrium_collection, field_stride)
    add_hydrodynamic_field(
        cache,
        temperature_material,
        materials["freezeout"],
        field_collection,
        surface_collection,
        qgp_collection,
        materials["qgp_core"],
        field_stride,
    )
    release_frames = add_particles(cache, materials, particle_collection)
    camera = add_camera_and_lights()
    add_stage_labels(cache, camera, materials["text"], release_frames)
    add_time_diagnostics(cache, camera, materials["text"])
    add_scene_metadata(scene, cache_path, cache)
    audit_scene(scene, cache)
    scene.frame_set(FRAME_HYDRO_START + 45)
    return scene, cache, release_frames


def render_previews(scene: bpy.types.Scene, cache: np.lib.npyio.NpzFile, output_dir: Path) -> None:
    particle_tau = cache["particle_tau"]
    hydro_tau = cache["hydro_tau"]
    median_particle_frame = tau_to_frame(float(np.median(particle_tau)), hydro_tau)
    previews = (
        (FRAME_CONTACT, "blender_physics_contact.png"),
        (FRAME_PREEQUILIBRIUM + 6, "blender_physics_qgp_formation.png"),
        (FRAME_HYDRO_START - 1, "blender_physics_landau_handoff.png"),
        (FRAME_HYDRO_START, "blender_physics_qgp_onset.png"),
        (FRAME_HYDRO_START + 42, "blender_physics_hydro.png"),
        (median_particle_frame, "blender_physics_particlization.png"),
        (FRAME_END, "blender_physics_final_hadrons.png"),
    )
    for frame, filename in previews:
        scene.frame_set(int(frame))
        scene.render.filepath = str((output_dir / filename).resolve())
        bpy.ops.render.render(write_still=True)
        print(f"Rendered preview: {output_dir / filename}")


def render_animation(scene: bpy.types.Scene, output_dir: Path) -> None:
    output = (output_dir / "hic_blender_physics_simulation.mp4").resolve()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; it is required to encode the animation.")
    with tempfile.TemporaryDirectory(prefix="hic_blender_frames_", dir=output_dir) as temporary_directory:
        frame_prefix = Path(temporary_directory) / "frame_"
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(frame_prefix)
        bpy.ops.render.render(animation=True)
        subprocess.run(
            (
                ffmpeg,
                "-y",
                "-framerate",
                str(FPS),
                "-start_number",
                str(FRAME_START),
                "-i",
                str(Path(temporary_directory) / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ),
            check=True,
        )
    print(f"Rendered animation: {output}")


def main() -> None:
    args = parse_args()
    scene, cache, _ = build_scene(args.cache, max(args.field_stride, 1))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Saved Blender scene: {output}")
    if not args.no_previews:
        render_previews(scene, cache, output.parent)
    if args.render_animation:
        render_animation(scene, output.parent)
    scene.frame_set(FRAME_HYDRO_START + 45)


if __name__ == "__main__":
    main()
