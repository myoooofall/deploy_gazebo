#!/usr/bin/env python3
import os
import sys
import threading
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D


TOPIC = "/nav_goal_body"


def _parse_xyz(text: str) -> Optional[Tuple[float, float, float]]:
    parts = text.strip().split()
    if len(parts) != 3:
        return None
    try:
        x, y, yaw = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None
    return x, y, yaw


class NavGoalPublisher(Node):
    def __init__(self) -> None:
        super().__init__("nav_goal_dialog")
        self.publisher = self.create_publisher(Pose2D, TOPIC, 10)

    def publish_goal(self, x: float, y: float, yaw: float) -> None:
        msg = Pose2D()
        msg.x = float(x)
        msg.y = float(y)
        msg.theta = float(yaw)
        self.publisher.publish(msg)
        self.get_logger().info(f"Publish {TOPIC}: x={msg.x:.3f} y={msg.y:.3f} yaw={msg.theta:.3f}")


def run_cli(node: NavGoalPublisher) -> int:
    print(f"Publishing body-frame goals to {TOPIC} (geometry_msgs/Pose2D)")
    print("Input format: x y yaw")
    print("Commands: 'q' to quit, 'z' to publish 0 0 0")
    while rclpy.ok():
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break
        if line.lower() in ("z", "zero", "clear"):
            node.publish_goal(0.0, 0.0, 0.0)
            continue
        parsed = _parse_xyz(line)
        if parsed is None:
            print("Invalid input. Expected: x y yaw (e.g. '2.0 0.0 0.0')")
            continue
        node.publish_goal(*parsed)
    return 0


def run_tk(node: NavGoalPublisher) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
        from tkinter import messagebox
    except Exception:
        return run_cli(node)

    # If there is no display, fall back to CLI only when running interactively.
    if sys.platform != "win32" and not os.environ.get("DISPLAY"):
        if sys.stdin is not None and sys.stdin.isatty():
            return run_cli(node)
        print(f"No DISPLAY and no TTY stdin; cannot run interactive goal dialog. Topic: {TOPIC}")
        return 0

    root = tk.Tk()
    root.title("RL_SAR Nav Goal Publisher")

    main = ttk.Frame(root, padding=12)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    x_var = tk.StringVar(value="2.0")
    y_var = tk.StringVar(value="0.0")
    yaw_var = tk.StringVar(value="0.0")
    status_var = tk.StringVar(value=f"Topic: {TOPIC}")

    def publish_clicked() -> None:
        try:
            x = float(x_var.get().strip())
            y = float(y_var.get().strip())
            yaw = float(yaw_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid input", "x, y, yaw must be numbers.")
            return
        node.publish_goal(x, y, yaw)
        status_var.set(f"Published: x={x:.3f} y={y:.3f} yaw={yaw:.3f}")

    def zero_clicked() -> None:
        x_var.set("0.0")
        y_var.set("0.0")
        yaw_var.set("0.0")
        publish_clicked()

    ttk.Label(main, text="Body-frame goal (x forward, y left, yaw rad)").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
    )

    ttk.Label(main, text="x").grid(row=1, column=0, sticky="w")
    ttk.Entry(main, textvariable=x_var, width=16).grid(row=1, column=1, sticky="ew", pady=2)
    ttk.Label(main, text="y").grid(row=2, column=0, sticky="w")
    ttk.Entry(main, textvariable=y_var, width=16).grid(row=2, column=1, sticky="ew", pady=2)
    ttk.Label(main, text="yaw").grid(row=3, column=0, sticky="w")
    ttk.Entry(main, textvariable=yaw_var, width=16).grid(row=3, column=1, sticky="ew", pady=2)

    buttons = ttk.Frame(main)
    buttons.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)
    buttons.columnconfigure(2, weight=1)

    ttk.Button(buttons, text="Publish", command=publish_clicked).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ttk.Button(buttons, text="Zero", command=zero_clicked).grid(row=0, column=1, sticky="ew", padx=6)
    ttk.Button(buttons, text="Quit", command=root.destroy).grid(row=0, column=2, sticky="ew", padx=(6, 0))

    ttk.Separator(main).grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
    ttk.Label(main, textvariable=status_var).grid(row=6, column=0, columnspan=2, sticky="w")

    main.columnconfigure(1, weight=1)

    root.mainloop()
    return 0


def main() -> int:
    rclpy.init()
    node = NavGoalPublisher()

    # Keep rclpy alive for logging, but we don't need to spin for publishing.
    spin_done = threading.Event()

    def _spin() -> None:
        while rclpy.ok() and not spin_done.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()

    try:
        return run_tk(node)
    finally:
        spin_done.set()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
