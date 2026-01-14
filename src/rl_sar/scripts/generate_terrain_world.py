#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from xml.etree.ElementTree import Element, SubElement, tostring


def _indent(elem: Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _add_material(visual: Element, gazebo_material: str) -> None:
    material = SubElement(visual, "material")
    script = SubElement(material, "script")
    uri = SubElement(script, "uri")
    uri.text = "file://media/materials/scripts/gazebo.material"
    name = SubElement(script, "name")
    name.text = gazebo_material


def _add_box_link(model: Element, name: str, pose_xyzrpy: Tuple[float, float, float, float, float, float],
                  size_xyz: Tuple[float, float, float], gazebo_material: str) -> None:
    link = SubElement(model, "link", {"name": name})
    pose = SubElement(link, "pose")
    pose.text = f"{pose_xyzrpy[0]} {pose_xyzrpy[1]} {pose_xyzrpy[2]} {pose_xyzrpy[3]} {pose_xyzrpy[4]} {pose_xyzrpy[5]}"

    collision = SubElement(link, "collision", {"name": f"{name}_collision"})
    geometry = SubElement(collision, "geometry")
    box = SubElement(geometry, "box")
    size = SubElement(box, "size")
    size.text = f"{size_xyz[0]} {size_xyz[1]} {size_xyz[2]}"

    visual = SubElement(link, "visual", {"name": f"{name}_visual"})
    geometry_v = SubElement(visual, "geometry")
    box_v = SubElement(geometry_v, "box")
    size_v = SubElement(box_v, "size")
    size_v.text = f"{size_xyz[0]} {size_xyz[1]} {size_xyz[2]}"
    _add_material(visual, gazebo_material)


def _add_cylinder_link(model: Element, name: str, pose_xyzrpy: Tuple[float, float, float, float, float, float],
                       radius: float, length: float, gazebo_material: str) -> None:
    link = SubElement(model, "link", {"name": name})
    pose = SubElement(link, "pose")
    pose.text = f"{pose_xyzrpy[0]} {pose_xyzrpy[1]} {pose_xyzrpy[2]} {pose_xyzrpy[3]} {pose_xyzrpy[4]} {pose_xyzrpy[5]}"

    collision = SubElement(link, "collision", {"name": f"{name}_collision"})
    geometry = SubElement(collision, "geometry")
    cyl = SubElement(geometry, "cylinder")
    r = SubElement(cyl, "radius")
    r.text = f"{radius}"
    l = SubElement(cyl, "length")
    l.text = f"{length}"

    visual = SubElement(link, "visual", {"name": f"{name}_visual"})
    geometry_v = SubElement(visual, "geometry")
    cyl_v = SubElement(geometry_v, "cylinder")
    r_v = SubElement(cyl_v, "radius")
    r_v.text = f"{radius}"
    l_v = SubElement(cyl_v, "length")
    l_v.text = f"{length}"
    _add_material(visual, gazebo_material)


@dataclass(frozen=True)
class Obstacle:
    kind: str  # "box" | "cylinder"
    name: str
    pose: Tuple[float, float, float, float, float, float]
    size: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    length: Optional[float] = None
    material: str = "Gazebo/Grey"


def _world_to_grid(x: float, y: float, size_m: float, res: float) -> Tuple[int, int]:
    half = size_m / 2.0
    gx = int((x + half) / res)
    gy = int((y + half) / res)
    return gx, gy


def _mark_box(occ: np.ndarray, size_m: float, res: float,
              cx: float, cy: float, sx: float, sy: float, margin: float) -> None:
    half = size_m / 2.0
    x1 = cx - sx / 2.0 - margin
    x2 = cx + sx / 2.0 + margin
    y1 = cy - sy / 2.0 - margin
    y2 = cy + sy / 2.0 + margin
    gx1 = max(0, int((x1 + half) / res))
    gx2 = min(occ.shape[1], int(math.ceil((x2 + half) / res)))
    gy1 = max(0, int((y1 + half) / res))
    gy2 = min(occ.shape[0], int(math.ceil((y2 + half) / res)))
    if gx1 < gx2 and gy1 < gy2:
        occ[gy1:gy2, gx1:gx2] = 1


def _mark_cylinder(occ: np.ndarray, size_m: float, res: float,
                   cx: float, cy: float, radius: float, margin: float) -> None:
    half = size_m / 2.0
    r = radius + margin
    gx1 = max(0, int(((cx - r) + half) / res))
    gx2 = min(occ.shape[1], int(math.ceil(((cx + r) + half) / res)))
    gy1 = max(0, int(((cy - r) + half) / res))
    gy2 = min(occ.shape[0], int(math.ceil(((cy + r) + half) / res)))
    if gx1 >= gx2 or gy1 >= gy2:
        return
    yy, xx = np.ogrid[gy1:gy2, gx1:gx2]
    wx = (xx + 0.5) * res - half
    wy = (yy + 0.5) * res - half
    mask = (wx - cx) ** 2 + (wy - cy) ** 2 <= r ** 2
    occ[gy1:gy2, gx1:gx2][mask] = 1


def _bfs_reachable_fraction(occ: np.ndarray, start_xy: Tuple[float, float], size_m: float, res: float) -> float:
    h, w = occ.shape
    sx, sy = start_xy
    gx, gy = _world_to_grid(sx, sy, size_m, res)
    gx = int(np.clip(gx, 0, w - 1))
    gy = int(np.clip(gy, 0, h - 1))
    if occ[gy, gx] != 0:
        return 0.0
    seen = np.zeros_like(occ, dtype=np.uint8)
    qx = [gx]
    qy = [gy]
    seen[gy, gx] = 1
    head = 0
    while head < len(qx):
        x = qx[head]
        y = qy[head]
        head += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and occ[ny, nx] == 0:
                seen[ny, nx] = 1
                qx.append(nx)
                qy.append(ny)
    free = float(np.sum(occ == 0))
    if free <= 1e-6:
        return 0.0
    return float(np.sum(seen)) / free


def _spawn_xy_for_corner(size_m: float, corner: str, inset: float = 0.9) -> Tuple[float, float]:
    half = size_m / 2.0
    x = -half + inset
    y = -half + inset
    if corner == "br":
        x = half - inset
        y = -half + inset
    elif corner == "tl":
        x = -half + inset
        y = half - inset
    elif corner == "tr":
        x = half - inset
        y = half - inset
    return x, y


def _spawn_sample_region_for_corner(size_m: float,
                                    corner: str,
                                    wall_thickness: float,
                                    spawn_keepout: float,
                                    spawn_box: float) -> Tuple[float, float, float, float]:
    """Return an axis-aligned sampling box (xmin, xmax, ymin, ymax) inside the fence."""
    half = size_m / 2.0
    inset = float(max(0.0, wall_thickness) + max(0.0, spawn_keepout))
    spawn_box = float(max(0.1, spawn_box))

    xmin_inner = -half + inset
    xmax_inner = half - inset
    ymin_inner = -half + inset
    ymax_inner = half - inset

    if xmin_inner >= xmax_inner or ymin_inner >= ymax_inner:
        # Inset too large; fall back to a small region near center.
        return -0.05, 0.05, -0.05, 0.05

    if corner == "bl":
        xmin = xmin_inner
        xmax = min(xmin_inner + spawn_box, xmax_inner)
        ymin = ymin_inner
        ymax = min(ymin_inner + spawn_box, ymax_inner)
    elif corner == "br":
        xmax = xmax_inner
        xmin = max(xmax_inner - spawn_box, xmin_inner)
        ymin = ymin_inner
        ymax = min(ymin_inner + spawn_box, ymax_inner)
    elif corner == "tl":
        xmin = xmin_inner
        xmax = min(xmin_inner + spawn_box, xmax_inner)
        ymax = ymax_inner
        ymin = max(ymax_inner - spawn_box, ymin_inner)
    elif corner == "tr":
        xmax = xmax_inner
        xmin = max(xmax_inner - spawn_box, xmin_inner)
        ymax = ymax_inner
        ymin = max(ymax_inner - spawn_box, ymin_inner)
    else:
        # Default: sample near center.
        xmin = max(-spawn_box / 2.0, xmin_inner)
        xmax = min(spawn_box / 2.0, xmax_inner)
        ymin = max(-spawn_box / 2.0, ymin_inner)
        ymax = min(spawn_box / 2.0, ymax_inner)

    return float(xmin), float(xmax), float(ymin), float(ymax)


def _sample_spawn_xy(rng: np.random.RandomState,
                     size_m: float,
                     corner: str,
                     wall_thickness: float,
                     spawn_keepout: float,
                     spawn_box: float,
                     max_tries: int = 128) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax = _spawn_sample_region_for_corner(
        size_m=size_m,
        corner=corner,
        wall_thickness=wall_thickness,
        spawn_keepout=spawn_keepout,
        spawn_box=spawn_box,
    )
    if xmax <= xmin or ymax <= ymin:
        return float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0)

    for _ in range(max_tries):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        return x, y
    return float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0)


def _yaw_towards_center(x: float, y: float) -> float:
    return float(math.atan2(-y, -x))


def _point_to_aabb_distance(px: float, py: float, cx: float, cy: float, sx: float, sy: float) -> float:
    dx = abs(px - cx) - sx / 2.0
    dy = abs(py - cy) - sy / 2.0
    dx = max(dx, 0.0)
    dy = max(dy, 0.0)
    return float(math.hypot(dx, dy))


def _is_outer_fence_name(name: str) -> bool:
    return name.startswith("outer_x_") or name.startswith("outer_y_")


def _obstacle_intersects_rect(ob: Obstacle, xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
    """Check whether obstacle footprint intersects a rectangle in XY."""
    if ob.kind == "box":
        assert ob.size is not None
        oxmin = ob.pose[0] - ob.size[0] / 2.0
        oxmax = ob.pose[0] + ob.size[0] / 2.0
        oymin = ob.pose[1] - ob.size[1] / 2.0
        oymax = ob.pose[1] + ob.size[1] / 2.0
        return (oxmin < xmax) and (oxmax > xmin) and (oymin < ymax) and (oymax > ymin)
    if ob.kind == "cylinder":
        assert ob.radius is not None
        # Circle-rectangle intersection.
        cx, cy = ob.pose[0], ob.pose[1]
        nx = min(max(cx, xmin), xmax)
        ny = min(max(cy, ymin), ymax)
        return (cx - nx) ** 2 + (cy - ny) ** 2 <= (ob.radius ** 2)
    return False


def _filter_obstacles_clear_spawn_region(obstacles: List[Obstacle],
                                        clear_rect: Tuple[float, float, float, float],
                                        keepout: float) -> List[Obstacle]:
    """Remove obstacles (except outer fence) that enter the spawn region plus keepout margin."""
    xmin, xmax, ymin, ymax = clear_rect
    margin = float(max(0.0, keepout))
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin
    out: List[Obstacle] = []
    for ob in obstacles:
        if _is_outer_fence_name(ob.name):
            out.append(ob)
            continue
        if _obstacle_intersects_rect(ob, xmin, xmax, ymin, ymax):
            continue
        out.append(ob)
    return out


def _generate_maze_obstacles(seed: int, difficulty: float, start_corner: str,
                             size_m: float, wall_thickness: float, wall_height: float,
                             cell_size_m: Optional[float],
                             spawn_keepout: float,
                             spawn_xy: Tuple[float, float]) -> Tuple[List[Obstacle], Tuple[float, float]]:
    rng = np.random.RandomState(seed)
    half = size_m / 2.0
    t = wall_thickness
    h = wall_height

    if cell_size_m is None:
        cell_size_m = float(np.clip(2.5 - difficulty * 1.0, 1.5, 2.5))

    inner = size_m - 2.0 * t
    cols = max(3, int(inner / cell_size_m))
    rows = max(3, int(inner / cell_size_m))
    cell_w = inner / cols
    cell_h = inner / rows

    # walls[r,c] = [N,E,S,W]
    walls = np.ones((rows, cols, 4), dtype=np.uint8)
    visited = np.zeros((rows, cols), dtype=np.uint8)

    def dfs(r: int, c: int) -> None:
        visited[r, c] = 1
        dirs = [(-1, 0, 0, 2), (0, 1, 1, 3), (1, 0, 2, 0), (0, -1, 3, 1)]
        rng.shuffle(dirs)
        for dr, dc, wi, wj in dirs:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < rows and 0 <= cc < cols and not visited[rr, cc]:
                walls[r, c, wi] = 0
                walls[rr, cc, wj] = 0
                dfs(rr, cc)

    dfs(int(rng.randint(0, rows)), int(rng.randint(0, cols)))

    obstacles: List[Obstacle] = []

    # Outer fence
    z = h / 2.0
    obstacles.append(Obstacle("box", "outer_x_pos", (half - t / 2.0, 0.0, z, 0, 0, 0), (t, size_m + t, h), material="Gazebo/Grey"))
    obstacles.append(Obstacle("box", "outer_x_neg", (-half + t / 2.0, 0.0, z, 0, 0, 0), (t, size_m + t, h), material="Gazebo/Grey"))
    obstacles.append(Obstacle("box", "outer_y_pos", (0.0, half - t / 2.0, z, 0, 0, 0), (size_m + t, t, h), material="Gazebo/Grey"))
    obstacles.append(Obstacle("box", "outer_y_neg", (0.0, -half + t / 2.0, z, 0, 0, 0), (size_m + t, t, h), material="Gazebo/Grey"))

    # Internal walls, unique edges (E and N only)
    x0 = -half + t
    y0 = -half + t

    clear_r = float(spawn_keepout)
    for r in range(rows):
        for c in range(cols):
            cx = x0 + (c + 0.5) * cell_w
            cy = y0 + (r + 0.5) * cell_h

            # East wall at x = x0 + (c+1)*cell_w
            if walls[r, c, 1] and c < cols - 1:
                wx = x0 + (c + 1) * cell_w
                wy = cy
                if (wx - spawn_xy[0]) ** 2 + (wy - spawn_xy[1]) ** 2 > clear_r ** 2:
                    obstacles.append(Obstacle(
                        "box",
                        f"maze_v_{r}_{c}",
                        (wx, wy, z, 0, 0, 0),
                        (t, cell_h + t, h),
                        material="Gazebo/Blue",
                    ))

            # Wall between row r and r+1:
            # Note: DFS indexes assume row increases "downwards", but our world Y increases "upwards".
            # So the boundary between (r,c) and (r+1,c) corresponds to wall index 2, not 0.
            if walls[r, c, 2] and r < rows - 1:
                wx = cx
                wy = y0 + (r + 1) * cell_h
                if (wx - spawn_xy[0]) ** 2 + (wy - spawn_xy[1]) ** 2 > clear_r ** 2:
                    obstacles.append(Obstacle(
                        "box",
                        f"maze_h_{r}_{c}",
                        (wx, wy, z, 0, 0, 0),
                        (cell_w + t, t, h),
                        material="Gazebo/Orange",
                    ))

    return obstacles, spawn_xy


def _generate_navigation_obstacles(seed: int, difficulty: float, start_corner: str,
                                   size_m: float, wall_thickness: float, wall_height: float,
                                   spawn_keepout: float,
                                   spawn_xy: Tuple[float, float]) -> Tuple[List[Obstacle], Tuple[float, float]]:
    rng = np.random.RandomState(seed)
    half = size_m / 2.0
    t = wall_thickness
    h = wall_height
    z = h / 2.0

    num_modules = int(4 + 30 * float(np.clip(difficulty, 0.0, 1.0)))
    open_alley_ratio = float(np.clip(0.7 - 0.5 * difficulty, 0.2, 0.9))
    min_spacing = 0.2

    res = 0.10  # occupancy resolution (m)
    occ = np.zeros((int(size_m / res), int(size_m / res)), dtype=np.uint8)
    obstacles: List[Obstacle] = []

    # Outer fence
    fence = [
        Obstacle("box", "outer_x_pos", (half - t / 2.0, 0.0, z, 0, 0, 0), (t, size_m + t, h), material="Gazebo/Grey"),
        Obstacle("box", "outer_x_neg", (-half + t / 2.0, 0.0, z, 0, 0, 0), (t, size_m + t, h), material="Gazebo/Grey"),
        Obstacle("box", "outer_y_pos", (0.0, half - t / 2.0, z, 0, 0, 0), (size_m + t, t, h), material="Gazebo/Grey"),
        Obstacle("box", "outer_y_neg", (0.0, -half + t / 2.0, z, 0, 0, 0), (size_m + t, t, h), material="Gazebo/Grey"),
    ]
    obstacles.extend(fence)
    for ob in fence:
        _mark_box(occ, size_m, res, ob.pose[0], ob.pose[1], ob.size[0], ob.size[1], margin=0.0)

    # Spawn safety zone (grid-based, plus a continuous keepout check in can_place_*).
    _mark_cylinder(occ, size_m, res, spawn_xy[0], spawn_xy[1], radius=float(spawn_keepout), margin=0.0)

    def can_place_box(cx: float, cy: float, sx: float, sy: float, margin: float) -> bool:
        if _point_to_aabb_distance(spawn_xy[0], spawn_xy[1], cx, cy, sx, sy) < float(spawn_keepout):
            return False
        tmp = np.zeros_like(occ)
        _mark_box(tmp, size_m, res, cx, cy, sx, sy, margin)
        return not np.any((tmp == 1) & (occ == 1))

    def can_place_cyl(cx: float, cy: float, radius: float, margin: float) -> bool:
        if float(math.hypot(cx - spawn_xy[0], cy - spawn_xy[1])) < float(spawn_keepout) + float(radius):
            return False
        tmp = np.zeros_like(occ)
        _mark_cylinder(tmp, size_m, res, cx, cy, radius, margin)
        return not np.any((tmp == 1) & (occ == 1))

    def commit_box(name: str, cx: float, cy: float, sx: float, sy: float, hh: float, mat: str) -> None:
        obstacles.append(Obstacle("box", name, (cx, cy, hh / 2.0, 0, 0, 0), (sx, sy, hh), material=mat))
        _mark_box(occ, size_m, res, cx, cy, sx, sy, margin=min_spacing)

    def commit_pillar(name: str, cx: float, cy: float, width: float, hh: float, mat: str) -> None:
        obstacles.append(Obstacle("box", name, (cx, cy, hh / 2.0, 0, 0, 0), (width, width, hh), material=mat))
        _mark_box(occ, size_m, res, cx, cy, width, width, margin=min_spacing)

    # Trap walls (teeth) along the fence, like in your training terrain.
    trap_gap = 3.0
    trap_depth = 0.8
    cur = -half + t + trap_gap / 2.0
    idx = 0
    while cur < half - t - trap_gap / 2.0:
        # bottom teeth (pointing +y)
        bx = cur
        by = -half + t + trap_depth / 2.0
        if can_place_box(bx, by, t, trap_depth, margin=min_spacing):
            commit_box(f"trap_b_{idx}", bx, by, t, trap_depth, h, "Gazebo/Green")
        # top teeth (pointing -y)
        tx = cur
        ty = half - t - trap_depth / 2.0
        if can_place_box(tx, ty, t, trap_depth, margin=min_spacing):
            commit_box(f"trap_t_{idx}", tx, ty, t, trap_depth, h, "Gazebo/Green")
        cur += trap_gap
        idx += 1
    cur = -half + t + trap_gap / 2.0
    idx = 0
    while cur < half - t - trap_gap / 2.0:
        # left teeth (pointing +x)
        lx = -half + t + trap_depth / 2.0
        ly = cur
        if can_place_box(lx, ly, trap_depth, t, margin=min_spacing):
            commit_box(f"trap_l_{idx}", lx, ly, trap_depth, t, h, "Gazebo/Green")
        # right teeth (pointing -x)
        rx = half - t - trap_depth / 2.0
        ry = cur
        if can_place_box(rx, ry, trap_depth, t, margin=min_spacing):
            commit_box(f"trap_r_{idx}", rx, ry, trap_depth, t, h, "Gazebo/Green")
        cur += trap_gap
        idx += 1

    # Random modules: pillar / alley / corner
    for i in range(num_modules):
        u = float(rng.rand())
        if u < 0.4:
            module = "pillar"
        elif u < 0.7:
            module = "alley"
        else:
            module = "corner"

        cx = float(rng.uniform(-half + 1.2, half - 1.2))
        cy = float(rng.uniform(-half + 1.2, half - 1.2))

        if module == "pillar":
            width = 0.6
            if can_place_box(cx, cy, width, width, margin=min_spacing):
                commit_pillar(f"pillar_{i}", cx, cy, width, h, "Gazebo/Black")
        elif module == "corner":
            arm_a = float(rng.uniform(1.0, 3.0))
            arm_b = float(rng.uniform(1.0, 3.0))
            # Two thin walls meeting at (cx, cy), biased by spawn corner like the IsaacGym terrain.
            if start_corner == "bl":
                rects = [(cx - arm_a / 2.0, cy, arm_a, t), (cx, cy - arm_b / 2.0, t, arm_b)]
            elif start_corner == "br":
                rects = [(cx + arm_a / 2.0, cy, arm_a, t), (cx, cy - arm_b / 2.0, t, arm_b)]
            elif start_corner == "tl":
                rects = [(cx - arm_a / 2.0, cy, arm_a, t), (cx, cy + arm_b / 2.0, t, arm_b)]
            elif start_corner == "tr":
                rects = [(cx + arm_a / 2.0, cy, arm_a, t), (cx, cy + arm_b / 2.0, t, arm_b)]
            else:
                rects = [(cx + arm_a / 2.0, cy, arm_a, t), (cx, cy + arm_b / 2.0, t, arm_b)]
            ok = all(can_place_box(x, y, sx, sy, margin=min_spacing) for (x, y, sx, sy) in rects)
            if ok:
                commit_box(f"corner_a_{i}", rects[0][0], rects[0][1], rects[0][2], rects[0][3], h, "Gazebo/Orange")
                commit_box(f"corner_b_{i}", rects[1][0], rects[1][1], rects[1][2], rects[1][3], h, "Gazebo/Orange")
        else:  # alley
            length = float(rng.uniform(1.0, 3.0))
            gap = float(rng.uniform(0.8, 1.6))
            orient = int(rng.randint(0, 2))  # 0: along x, 1: along y
            is_open = (rng.rand() < open_alley_ratio)
            if orient == 0:
                w1 = (cx, cy - gap / 2.0 - t / 2.0, length, t)
                w2 = (cx, cy + gap / 2.0 + t / 2.0, length, t)
                rects = [w1, w2]
                if not is_open:
                    # Bias which end gets capped so that, statistically, the corridor opens towards the spawn corner.
                    if start_corner in ("bl", "tl"):
                        cap = (cx + length / 2.0 - t / 2.0, cy, t, gap + 2 * t)  # cap +x end
                    elif start_corner in ("br", "tr"):
                        cap = (cx - length / 2.0 + t / 2.0, cy, t, gap + 2 * t)  # cap -x end
                    else:
                        cap = (cx + length / 2.0 - t / 2.0, cy, t, gap + 2 * t)
                    rects.append(cap)
            else:
                w1 = (cx - gap / 2.0 - t / 2.0, cy, t, length)
                w2 = (cx + gap / 2.0 + t / 2.0, cy, t, length)
                rects = [w1, w2]
                if not is_open:
                    if start_corner in ("bl", "br"):
                        cap = (cx, cy + length / 2.0 - t / 2.0, gap + 2 * t, t)  # cap +y end
                    elif start_corner in ("tl", "tr"):
                        cap = (cx, cy - length / 2.0 + t / 2.0, gap + 2 * t, t)  # cap -y end
                    else:
                        cap = (cx, cy + length / 2.0 - t / 2.0, gap + 2 * t, t)
                    rects.append(cap)
            ok = all(can_place_box(x, y, sx, sy, margin=min_spacing) for (x, y, sx, sy) in rects)
            if ok:
                for k, (x, y, sx, sy) in enumerate(rects):
                    commit_box(f"alley_{i}_{k}", x, y, sx, sy, h, "Gazebo/Blue")

    return obstacles, spawn_xy


def _build_world_xml(world_name: str, obstacles: List[Obstacle]) -> bytes:
    sdf = Element("sdf", {"version": "1.5"})
    world = SubElement(sdf, "world", {"name": world_name})

    physics = SubElement(world, "physics", {"type": "ode"})
    SubElement(physics, "max_step_size").text = "0.0005"
    SubElement(physics, "real_time_factor").text = "1"
    SubElement(physics, "real_time_update_rate").text = "2000"
    SubElement(physics, "gravity").text = "0 0 -9.81"
    ode = SubElement(physics, "ode")
    solver = SubElement(ode, "solver")
    SubElement(solver, "type").text = "quick"
    SubElement(solver, "iters").text = "50"
    SubElement(solver, "sor").text = "1.3"
    constraints = SubElement(ode, "constraints")
    SubElement(constraints, "cfm").text = "0.0"
    SubElement(constraints, "erp").text = "0.2"
    SubElement(constraints, "contact_max_correcting_vel").text = "10.0"
    SubElement(constraints, "contact_surface_layer").text = "0.001"

    scene = SubElement(world, "scene")
    sky = SubElement(scene, "sky")
    clouds = SubElement(sky, "clouds")
    SubElement(clouds, "speed").text = "12"

    include_sun = SubElement(world, "include")
    SubElement(include_sun, "uri").text = "model://sun"
    include_ground = SubElement(world, "include")
    SubElement(include_ground, "uri").text = "model://ground_plane"

    env_model = SubElement(world, "model", {"name": "static_environment"})
    SubElement(env_model, "static").text = "true"

    for ob in obstacles:
        if ob.kind == "box":
            assert ob.size is not None
            _add_box_link(env_model, ob.name, ob.pose, ob.size, ob.material)
        elif ob.kind == "cylinder":
            assert ob.radius is not None and ob.length is not None
            _add_cylinder_link(env_model, ob.name, ob.pose, ob.radius, ob.length, ob.material)

    _indent(sdf)
    return tostring(sdf, encoding="utf-8", xml_declaration=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["maze", "navigation"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--start-corner", choices=["bl", "br", "tl", "tr"], default="bl")
    ap.add_argument("--out", type=str, default="/tmp/rl_sar_generated.world")
    ap.add_argument("--size", type=float, default=12.0)
    ap.add_argument("--wall-thickness", type=float, default=0.3)
    ap.add_argument("--wall-height", type=float, default=0.5)
    ap.add_argument("--cell-size", type=float, default=None)
    ap.add_argument("--spawn-keepout", type=float, default=1.2)
    ap.add_argument("--spawn-box", type=float, default=1.0)
    ap.add_argument("--check-inflation", type=float, default=0.4)
    ap.add_argument("--min-reachable", type=float, default=0.98)
    ap.add_argument("--max-attempts", type=int, default=20)
    ap.add_argument("--print-json", action="store_true", help="Print JSON {world_path, spawn_x, spawn_y, spawn_yaw}.")
    args = ap.parse_args()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    seed = int(args.seed)
    attempt = 0
    last_spawn = (0.0, 0.0)
    last_obstacles: List[Obstacle] = []

    while attempt < args.max_attempts:
        rng_spawn = np.random.RandomState(seed + attempt + 1000003)
        spawn_xy = _sample_spawn_xy(
            rng=rng_spawn,
            size_m=float(args.size),
            corner=str(args.start_corner),
            wall_thickness=float(args.wall_thickness),
            spawn_keepout=float(args.spawn_keepout),
            spawn_box=float(args.spawn_box),
        )
        spawn_rect = _spawn_sample_region_for_corner(
            size_m=float(args.size),
            corner=str(args.start_corner),
            wall_thickness=float(args.wall_thickness),
            spawn_keepout=float(args.spawn_keepout),
            spawn_box=float(args.spawn_box),
        )

        if args.type == "maze":
            obs, spawn_xy = _generate_maze_obstacles(
                seed=seed + attempt,
                difficulty=float(args.difficulty),
                start_corner=args.start_corner,
                size_m=float(args.size),
                wall_thickness=float(args.wall_thickness),
                wall_height=float(args.wall_height),
                cell_size_m=args.cell_size,
                spawn_keepout=float(args.spawn_keepout),
                spawn_xy=spawn_xy,
            )
        else:
            obs, spawn_xy = _generate_navigation_obstacles(
                seed=seed + attempt,
                difficulty=float(args.difficulty),
                start_corner=args.start_corner,
                size_m=float(args.size),
                wall_thickness=float(args.wall_thickness),
                wall_height=float(args.wall_height),
                spawn_keepout=float(args.spawn_keepout),
                spawn_xy=spawn_xy,
            )

        # Reserve the entire spawn box region: do not allow any internal obstacles there.
        obs = _filter_obstacles_clear_spawn_region(
            obstacles=obs,
            clear_rect=spawn_rect,
            keepout=float(args.spawn_keepout),
        )

        # Continuous check: spawn must not be inside any obstacle footprint.
        spawn_ok = True
        for ob in obs:
            if ob.kind == "box":
                if abs(spawn_xy[0] - ob.pose[0]) <= (ob.size[0] / 2.0) and abs(spawn_xy[1] - ob.pose[1]) <= (ob.size[1] / 2.0):
                    spawn_ok = False
                    break
            else:
                if float(math.hypot(spawn_xy[0] - ob.pose[0], spawn_xy[1] - ob.pose[1])) <= float(ob.radius):
                    spawn_ok = False
                    break
        if not spawn_ok:
            attempt += 1
            continue

        # Validate reachability (simple 2D BFS on inflated occupancy of obstacles)
        size_m = float(args.size)
        res = 0.10
        occ = np.zeros((int(size_m / res), int(size_m / res)), dtype=np.uint8)
        for ob in obs:
            if ob.kind == "box":
                _mark_box(occ, size_m, res, ob.pose[0], ob.pose[1], ob.size[0], ob.size[1], margin=float(args.check_inflation))
            else:
                _mark_cylinder(occ, size_m, res, ob.pose[0], ob.pose[1], ob.radius, margin=float(args.check_inflation))
        frac = _bfs_reachable_fraction(occ, spawn_xy, size_m, res)
        last_spawn = spawn_xy
        last_obstacles = obs
        if frac >= float(args.min_reachable):
            break
        attempt += 1

    xml = _build_world_xml(f"{args.type}_terrain", last_obstacles)
    with open(out_path, "wb") as f:
        f.write(xml)
        f.write(b"\n")
        f.write(f"<!-- seed={args.seed} attempt={attempt} difficulty={args.difficulty} start_corner={args.start_corner} spawn_xy={last_spawn} -->\n".encode("utf-8"))

    if args.print_json:
        spawn_yaw = _yaw_towards_center(last_spawn[0], last_spawn[1])
        print(json.dumps({
            "world_path": out_path,
            "spawn_x": float(last_spawn[0]),
            "spawn_y": float(last_spawn[1]),
            "spawn_yaw": float(spawn_yaw),
            "seed": int(args.seed),
            "attempt": int(attempt),
            "difficulty": float(args.difficulty),
            "start_corner": str(args.start_corner),
        }))
    else:
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
