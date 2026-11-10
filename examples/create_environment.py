"""Interactive editor for custom obstacle-avoidance scenarios.

Place circle and box obstacles (and the start / goal markers) with the
mouse in a matplotlib window, and save the result as a JSON file that
the testbed can load with ``--scenario``::

    python examples/create_environment.py                   # blank 10x10
    python examples/create_environment.py my_scenario.json  # edit existing
    python examples/create_environment.py --out mine.json   # output name

Keys
----
c : place a circle  (click to place, drag to set the radius)
b : place a box     (drag from one corner to the opposite one)
d : delete          (click an obstacle)
m : move            (drag an obstacle by its center)
s : move the start marker
g : move the goal marker
F2: save,  q / Esc : quit

Obstacles are created static; to make one move, add a ``"velocity"``
field to its entry in the saved JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Rectangle as RectanglePatch
from reactive_avoidance_testbed import (
    BoxObstacle,
    CircleObstacle,
    _interactive_backend,
)

DEFAULT_RADIUS = 0.5
"""Radius of a newly placed circle before it is dragged."""

MIN_SIZE = 0.05
"""Smallest accepted radius / box extent; smaller placements are kept
at the default (circles) or discarded (boxes)."""


# ---------------------------------------------------------------------------
# Scenario data (JSON format)
# ---------------------------------------------------------------------------
def default_data() -> Dict[str, object]:
    """Blank 10x10 world with the demo start / goal."""
    return {
        "bounds": [[0.0, 10.0], [0.0, 10.0]],
        "start": [1.0, 1.0],
        "goal": [9.0, 9.0],
        "obstacles": [],
    }


def load_data(path: str) -> Dict[str, object]:
    """Read and sanity-check a scenario JSON file."""
    with open(path) as fh:
        data = json.load(fh)
    for key in ("bounds", "start", "goal", "obstacles"):
        if key not in data:
            raise SystemExit(f"scenario file {path!s}: missing key {key!r}")
    for entry in data["obstacles"]:
        if "type" not in entry:
            raise SystemExit(
                f"scenario file {path!s}: obstacle missing key 'type'"
            )
        if entry["type"] == "circle":
            required = ("center", "radius")
        elif entry["type"] == "box":
            required = ("center", "half_extents")
        else:
            raise SystemExit(
                f"scenario file {path!s}: unknown obstacle type "
                f"{entry['type']!r}"
            )
        for key in required:
            if key not in entry:
                raise SystemExit(
                    f"scenario file {path!s}: obstacle missing key {key!r}"
                )
    return data


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------
class ScenarioEditor:
    """Matplotlib window for placing obstacles with the mouse."""

    HINTS: Dict[str, str] = {
        "circle": "c: click to place, drag to set the radius",
        "box": "b: drag from one corner to the opposite one",
        "delete": "d: click an obstacle to remove it",
        "move": "m: drag an obstacle to reposition it",
        "start": "s: click to move the start marker",
        "goal": "g: click to move the goal marker",
    }

    def __init__(self, data: Dict[str, object], out_path: str) -> None:
        self.data = data
        self.out_path = out_path
        self.mode: str = "circle"
        # Free the keys the editor uses from matplotlib's default shortcuts
        # (s = save dialog, g = grid toggle, q = quit, c = back, ...).
        for rc_key in (
            "keymap.save", "keymap.grid", "keymap.quit", "keymap.back"
        ):
            plt.rcParams[rc_key] = [
                k for k in plt.rcParams[rc_key]
                if k not in ("s", "g", "q", "c")
            ]
        self.fig, self.ax = plt.subplots(figsize=(9.0, 9.0))
        self._press: Optional[np.ndarray] = None
        # obstacle entry (move), its index (circle), or 'start'/'goal'
        self._target: object = None
        self._orig_center: Optional[np.ndarray] = None
        self._temp: Optional[matplotlib.patches.Patch] = None
        self._redraw()
        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("key_press_event", self._on_key)

    # -- Drawing ------------------------------------------------------------
    def _redraw(self) -> None:
        """Redraw the whole figure (world, obstacles, markers, hints)."""
        self.ax.clear()
        data = self.data
        bounds = data["bounds"]
        self.ax.set_xlim(float(bounds[0][0]), float(bounds[0][1]))
        self.ax.set_ylim(float(bounds[1][0]), float(bounds[1][1]))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        for entry in data["obstacles"]:
            self._draw_obstacle(self.ax, entry)
        start = data["start"]
        self.ax.plot([start[0]], [start[1]], "o", color="tab:green", ms=11)
        self.ax.annotate(
            "start",
            (float(start[0]), float(start[1])),
            xytext=(6, 6),
            textcoords="offset points",
            color="tab:green",
        )
        goal = data["goal"]
        self.ax.plot([goal[0]], [goal[1]], "*", color="red", ms=15)
        self.ax.annotate(
            "goal",
            (float(goal[0]), float(goal[1])),
            xytext=(6, 6),
            textcoords="offset points",
            color="red",
        )
        self.ax.set_title(f"mode: {self.mode}   |   {self.out_path}")
        self.ax.text(
            0.02,
            0.02,
            self.HINTS[self.mode],
            transform=self.ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )
        self.fig.canvas.draw_idle()

    @staticmethod
    def _draw_obstacle(
        ax: matplotlib.axes.Axes, entry: Dict[str, object]
    ) -> None:
        if entry["type"] == "circle":
            patch = CirclePatch(
                (float(entry["center"][0]), float(entry["center"][1])),
                float(entry["radius"]),
                facecolor="lightgray",
                alpha=0.7,
                edgecolor="k",
            )
        else:
            lo = (
                float(entry["center"][0]) - float(entry["half_extents"][0]),
                float(entry["center"][1]) - float(entry["half_extents"][1]),
            )
            patch = RectanglePatch(
                lo,
                float(2.0 * entry["half_extents"][0]),
                float(2.0 * entry["half_extents"][1]),
                facecolor="lightgray",
                alpha=0.7,
                edgecolor="k",
            )
        ax.add_patch(patch)

    def _draw_temp(self, point: np.ndarray) -> None:
        """(Re)draw the in-progress box."""
        if self._temp is not None:
            self._temp.remove()
        lo = np.minimum(self._press, point)
        hi = np.maximum(self._press, point)
        self._temp = RectanglePatch(
            tuple(lo),
            float(hi[0] - lo[0]),
            float(hi[1] - lo[1]),
            fill=False,
            linestyle="--",
            color="tab:blue",
        )
        self.ax.add_patch(self._temp)
        self.fig.canvas.draw_idle()

    # -- Hit testing ----------------------------------------------------------
    def _hit(self, x: float, y: float) -> Optional[Dict[str, object]]:
        """Obstacle dict under (x, y), topmost first, or None."""
        point = np.array([x, y], dtype=float)
        for entry in reversed(self.data["obstacles"]):
            if entry["type"] == "circle":
                ob = CircleObstacle(entry["center"], float(entry["radius"]))
            else:
                ob = BoxObstacle(entry["center"], entry["half_extents"])
            if float(ob.signed_distance(point)[0]) <= 0.0:
                return entry
        return None

    # -- Events ---------------------------------------------------------------
    def _on_press(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if event.inaxes is not self.ax or event.button != 1:
            return
        toolbar = self.fig.canvas.toolbar
        if toolbar is not None and toolbar.mode:
            return  # pan / zoom tool is active
        x = float(event.xdata)
        y = float(event.ydata)
        self._press = np.array([x, y], dtype=float)
        if self.mode == "circle":
            self.data["obstacles"].append(
                {"type": "circle", "center": [x, y], "radius": DEFAULT_RADIUS}
            )
            self._target = len(self.data["obstacles"]) - 1
            self._redraw()
        elif self.mode == "delete":
            entry = self._hit(x, y)
            if entry is not None:
                self.data["obstacles"].remove(entry)
                self._redraw()
        elif self.mode == "move":
            entry = self._hit(x, y)
            if entry is not None:
                self._target = entry
                self._orig_center = np.asarray(
                    entry["center"], dtype=float
                ).copy()
        elif self.mode in ("start", "goal"):
            self.data[self.mode] = [x, y]
            self._target = self.mode
            self._redraw()

    def _on_motion(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if event.inaxes is not self.ax or self._press is None:
            return
        point = np.array(
            [float(event.xdata), float(event.ydata)], dtype=float
        )
        if self.mode == "circle":
            entry = self.data["obstacles"][self._target]
            entry["radius"] = max(
                float(np.linalg.norm(point - self._press)), MIN_SIZE
            )
            self._redraw()
        elif self.mode == "box":
            self._draw_temp(point)
        elif self.mode == "move":
            entry = self._target
            entry["center"] = [
                float(self._orig_center[0] + point[0] - self._press[0]),
                float(self._orig_center[1] + point[1] - self._press[1]),
            ]
            self._redraw()
        elif self.mode in ("start", "goal"):
            self.data[self.mode] = [float(point[0]), float(point[1])]
            self._redraw()

    def _on_release(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if (
            event.inaxes is not self.ax
            or event.button != 1
            or self._press is None
        ):
            return
        point = np.array(
            [float(event.xdata), float(event.ydata)], dtype=float
        )
        if self.mode == "circle":
            entry = self.data["obstacles"][self._target]
            radius = float(np.linalg.norm(point - self._press))
            if radius >= MIN_SIZE:
                entry["radius"] = radius
        elif self.mode == "box":
            if self._temp is not None:
                self._temp.remove()
                self._temp = None
            lo = np.minimum(self._press, point)
            hi = np.maximum(self._press, point)
            if float(np.linalg.norm(hi - lo)) >= MIN_SIZE:
                center = (lo + hi) * 0.5
                half = (hi - lo) / 2.0
                self.data["obstacles"].append(
                    {
                        "type": "box",
                        "center": [float(center[0]), float(center[1])],
                        "half_extents": [float(half[0]), float(half[1])],
                    }
                )
        self._press = None
        self._target = None
        self._orig_center = None
        self._redraw()

    def _on_key(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        key = event.key
        if key in ("c", "b", "d", "m", "s", "g"):
            self.mode = {
                "c": "circle",
                "b": "box",
                "d": "delete",
                "m": "move",
                "s": "start",
                "g": "goal",
            }[key]
            self._redraw()
        elif key == "f2":
            self.save()
        elif key in ("q", "escape"):
            plt.close(self.fig)

    # -- Saving ---------------------------------------------------------------
    def save(self) -> None:
        """Write the scenario to ``out_path`` (with sanity warnings)."""
        data = self.data
        start = np.asarray(data["start"], dtype=float)
        goal = np.asarray(data["goal"], dtype=float)
        obstacles = data["obstacles"]
        for name, pos in (("start", start), ("goal", goal)):
            for entry in obstacles:
                if entry["type"] == "circle":
                    ob = CircleObstacle(
                        entry["center"], float(entry["radius"])
                    )
                else:
                    ob = BoxObstacle(entry["center"], entry["half_extents"])
                if float(ob.signed_distance(pos)[0]) < 0.0:
                    print(
                        f"warning: {name} {np.round(pos, 3)} "
                        "lies inside an obstacle"
                    )
        out = {
            "bounds": [[float(v) for v in row] for row in data["bounds"]],
            "start": [round(float(v), 4) for v in start],
            "goal": [round(float(v), 4) for v in goal],
            "obstacles": [self._entry_json(entry) for entry in obstacles],
        }
        with open(self.out_path, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        print(
            f"scenario saved to {self.out_path} "
            f"({len(obstacles)} obstacles)"
        )

    @staticmethod
    def _entry_json(entry: Dict[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {
            "type": entry["type"],
            "center": [round(float(v), 4) for v in entry["center"]],
        }
        if entry["type"] == "circle":
            out["radius"] = round(float(entry["radius"]), 4)
        else:
            out["half_extents"] = [
                round(float(v), 4) for v in entry["half_extents"]
            ]
        if "velocity" in entry:
            out["velocity"] = [round(float(v), 4) for v in entry["velocity"]]
        return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a custom obstacle-avoidance scenario by "
        "placing obstacles with the mouse."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="existing scenario JSON to load and edit",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="JSON",
        help="output file (default: the input file, or "
        "custom_environment.json)",
    )
    args = parser.parse_args()
    if not _interactive_backend():
        print("an interactive display is required (no GUI detected)")
        sys.exit(1)
    data = load_data(args.input) if args.input is not None else default_data()
    if args.out is not None:
        out_path = args.out
    elif args.input is not None:
        out_path = args.input
    else:
        out_path = "custom_environment.json"
    print("Keys: c circle | b box | d delete | m move | s start | g goal")
    print("      F2 save | q / Esc quit")
    # Keep a reference: matplotlib only holds weak references to bound-method
    # callbacks, so an unreferenced editor is garbage-collected and every
    # mouse / key handler silently stops working.
    editor = ScenarioEditor(data, out_path)  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
