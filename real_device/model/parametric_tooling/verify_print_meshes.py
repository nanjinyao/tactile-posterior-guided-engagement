#!/usr/bin/env python3
"""Validate generated STL and 3MF topology with Open3D and write a JSON report."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import open3d as o3d


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "generated"


def read_3mf_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Read the single CadQuery mesh payload from a generated 3MF archive."""

    with ZipFile(path) as archive:
        model_names = [
            name for name in archive.namelist()
            if name.lower().endswith(".model")
        ]
        if len(model_names) != 1:
            raise ValueError(
                f"{path.name}: expected one 3MF model payload, got {model_names}"
            )
        root = ET.fromstring(archive.read(model_names[0]))
    namespace = root.tag.partition("}")[0].lstrip("{")
    mesh_node = root.find(f".//{{{namespace}}}mesh")
    if mesh_node is None:
        raise ValueError(f"{path.name}: no mesh object found")
    vertices_node = mesh_node.find(f"{{{namespace}}}vertices")
    triangles_node = mesh_node.find(f"{{{namespace}}}triangles")
    if vertices_node is None or triangles_node is None:
        raise ValueError(f"{path.name}: incomplete mesh payload")
    vertices = np.asarray(
        [
            [float(row.attrib[axis]) for axis in ("x", "y", "z")]
            for row in vertices_node
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [int(row.attrib[index]) for index in ("v1", "v2", "v3")]
            for row in triangles_node
        ],
        dtype=np.int32,
    )
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )


def main() -> None:
    records = []
    failed = []
    paths = sorted(OUTPUT_DIR.glob("*.stl")) + sorted(
        OUTPUT_DIR.glob("*.3mf")
    )
    for path in paths:
        if path.suffix.lower() == ".3mf":
            mesh = read_3mf_mesh(path)
        else:
            mesh = o3d.io.read_triangle_mesh(
                str(path),
                enable_post_processing=False,
            )
        mesh.merge_close_vertices(1.0e-5)
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        record = {
            "file": path.name,
            "format": path.suffix.lower().lstrip("."),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "watertight": bool(mesh.is_watertight()),
            "edge_manifold": bool(mesh.is_edge_manifold(allow_boundary_edges=False)),
            "vertex_manifold": bool(mesh.is_vertex_manifold()),
            "self_intersecting": bool(mesh.is_self_intersecting()),
            "orientable": bool(mesh.is_orientable()),
        }
        record["pass"] = (
            record["watertight"]
            and record["edge_manifold"]
            and record["vertex_manifold"]
            and not record["self_intersecting"]
            and record["orientable"]
        )
        records.append(record)
        if not record["pass"]:
            failed.append(path.name)

    report = {
        "validator": f"Open3D {o3d.__version__}",
        "validated_formats": ["stl", "3mf"],
        "merge_close_vertices_mm": 1.0e-5,
        "all_pass": not failed,
        "failed": failed,
        "meshes": records,
    }
    report_path = OUTPUT_DIR / "mesh_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit("Print-mesh validation failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
