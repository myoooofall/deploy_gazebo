#!/usr/bin/env python3
import math
import os
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class GoalRepublisher(Node):
    def __init__(self):
        super().__init__(
            "goal_republisher",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._goal_actual = None
        self._goal_pred = None
        self._last_odom = None
        self._latest_odom_state = None  # (stamp_ns, x, y, yaw_corr)
        self._odom_hist = deque(maxlen=5000)
        self._actual_goal_key = None
        self._actual_pending_reanchor = False
        self._actual_anchor_stamp_ns = None
        self._actual_latched_map_pose = None
        self._yaw_offset_deg = float(os.getenv("GOAL_VIZ_YAW_OFFSET_DEG", "-90.0"))
        self._yaw_offset_rad = math.radians(self._yaw_offset_deg)
        self._odom_match_tolerance_ns = int(0.30 * 1e9)  # 300ms

        self._pub_actual = self.create_publisher(PoseStamped, "/nav/goal_actual_map_viz", qos)
        self._pub_pred = self.create_publisher(PoseStamped, "/nav/goal_pred_map_viz", qos)
        self._pub_odom_viz = self.create_publisher(Odometry, "/odom_viz", qos)

        self.create_subscription(PoseStamped, "/nav/goal_actual_map", self._on_actual, qos)
        self.create_subscription(PoseStamped, "/nav/goal_pred_map", self._on_pred, qos)
        self.create_subscription(Odometry, "/odom", self._on_odom, qos)

        self.create_timer(0.1, self._republish)
        self.get_logger().info(f"goal_republisher yaw offset(deg)={self._yaw_offset_deg:.1f}")

    def _on_actual(self, msg: PoseStamped):
        self._goal_actual = msg
        key = self._goal_key(msg)
        if self._actual_goal_key is None or not self._same_goal_key(key, self._actual_goal_key):
            self._actual_goal_key = key
            self._actual_pending_reanchor = True
            self._actual_anchor_stamp_ns = self._stamp_to_ns(msg.header.stamp)

    def _on_pred(self, msg: PoseStamped):
        self._goal_pred = msg

    @staticmethod
    def _goal_key(msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        return (msg.header.frame_id, p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    @staticmethod
    def _same_goal_key(a, b, eps: float = 1e-4) -> bool:
        if a[0] != b[0]:
            return False
        for i in range(1, 8):
            if abs(a[i] - b[i]) > eps:
                return False
        return True

    @staticmethod
    def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quat_from_yaw(yaw: float):
        half = 0.5 * yaw
        return 0.0, 0.0, math.sin(half), math.cos(half)

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1000000000 + int(stamp.nanosec)

    def _on_odom(self, msg: Odometry):
        self._last_odom = msg
        stamp_ns = self._stamp_to_ns(msg.header.stamp)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw_corr = self._yaw_from_quat(q.x, q.y, q.z, q.w) + self._yaw_offset_rad
        odom_state = (stamp_ns, p.x, p.y, yaw_corr)
        self._latest_odom_state = odom_state
        self._odom_hist.append(odom_state)

    def _lookup_odom_by_stamp(self, target_ns: int):
        if target_ns is None:
            return self._latest_odom_state
        if not self._odom_hist:
            return self._latest_odom_state

        nearest = None
        nearest_abs_dt = None
        for item in self._odom_hist:
            dt = abs(item[0] - target_ns)
            if nearest is None or dt < nearest_abs_dt:
                nearest = item
                nearest_abs_dt = dt

        if nearest is not None and nearest_abs_dt is not None and nearest_abs_dt <= self._odom_match_tolerance_ns:
            return nearest
        return self._latest_odom_state

    def _to_map_pose(self, msg: PoseStamped, odom_state) -> PoseStamped:
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        if msg.header.frame_id == "map":
            out.header.frame_id = "map"
            out.pose = msg.pose
            return out

        if msg.header.frame_id != "base_link":
            out.header.frame_id = msg.header.frame_id
            out.pose = msg.pose
            return out

        if odom_state is None:
            return None

        _, base_map_x, base_map_y, base_map_yaw = odom_state
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        byaw = base_map_yaw
        cy = math.cos(byaw)
        sy = math.sin(byaw)

        out.header.frame_id = "map"
        out.pose.position.x = base_map_x + cy * gx - sy * gy
        out.pose.position.y = base_map_y + sy * gx + cy * gy
        out.pose.position.z = msg.pose.position.z

        gyaw = self._yaw_from_quat(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        qx, qy, qz, qw = self._quat_from_yaw(byaw + gyaw)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        return out

    def _republish(self):
        if self._last_odom is not None:
            out_odom = Odometry()
            out_odom.header.stamp = self.get_clock().now().to_msg()
            out_odom.header.frame_id = self._last_odom.header.frame_id
            out_odom.child_frame_id = self._last_odom.child_frame_id
            out_odom.pose.pose.position = self._last_odom.pose.pose.position
            q = self._last_odom.pose.pose.orientation
            yaw = self._yaw_from_quat(q.x, q.y, q.z, q.w) + self._yaw_offset_rad
            qx, qy, qz, qw = self._quat_from_yaw(yaw)
            out_odom.pose.pose.orientation.x = qx
            out_odom.pose.pose.orientation.y = qy
            out_odom.pose.pose.orientation.z = qz
            out_odom.pose.pose.orientation.w = qw
            out_odom.pose.covariance = self._last_odom.pose.covariance
            out_odom.twist = self._last_odom.twist
            self._pub_odom_viz.publish(out_odom)

        if self._goal_actual is not None:
            if self._goal_actual.header.frame_id == "map":
                out = PoseStamped()
                out.header.stamp = self.get_clock().now().to_msg()
                out.header.frame_id = "map"
                out.pose = self._goal_actual.pose
                self._pub_actual.publish(out)
            else:
                if self._actual_pending_reanchor:
                    odom_state = self._lookup_odom_by_stamp(self._actual_anchor_stamp_ns)
                    out = self._to_map_pose(self._goal_actual, odom_state)
                    if out is not None:
                        self._actual_latched_map_pose = out.pose
                        self._actual_pending_reanchor = False

                if self._actual_latched_map_pose is not None:
                    out = PoseStamped()
                    out.header.stamp = self.get_clock().now().to_msg()
                    out.header.frame_id = "map"
                    out.pose = self._actual_latched_map_pose
                    self._pub_actual.publish(out)

        if self._goal_pred is not None:
            pred_stamp_ns = self._stamp_to_ns(self._goal_pred.header.stamp)
            odom_state = self._lookup_odom_by_stamp(pred_stamp_ns)
            out = self._to_map_pose(self._goal_pred, odom_state)
            if out is not None:
                self._pub_pred.publish(out)


def main():
    rclpy.init()
    node = GoalRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
