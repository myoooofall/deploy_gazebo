#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped


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

        self._pub_actual = self.create_publisher(PoseStamped, "/nav/goal_actual_map_viz", qos)
        self._pub_pred = self.create_publisher(PoseStamped, "/nav/goal_pred_map_viz", qos)

        self.create_subscription(PoseStamped, "/nav/goal_actual_map", self._on_actual, qos)
        self.create_subscription(PoseStamped, "/nav/goal_pred_map", self._on_pred, qos)

        self.create_timer(0.2, self._republish)

    def _on_actual(self, msg: PoseStamped):
        self._goal_actual = msg

    def _on_pred(self, msg: PoseStamped):
        self._goal_pred = msg

    def _republish(self):
        now = self.get_clock().now().to_msg()
        if self._goal_actual is not None:
            out = PoseStamped()
            out.header.stamp = now
            out.header.frame_id = self._goal_actual.header.frame_id
            out.pose = self._goal_actual.pose
            self._pub_actual.publish(out)
        if self._goal_pred is not None:
            out = PoseStamped()
            out.header.stamp = now
            out.header.frame_id = self._goal_pred.header.frame_id
            out.pose = self._goal_pred.pose
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
