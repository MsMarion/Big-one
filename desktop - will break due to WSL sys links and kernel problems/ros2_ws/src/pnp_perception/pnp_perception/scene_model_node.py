"""Scene model from the depth camera: the objects on the table as measured boxes, compared with the truth.

    /perception/object_points (obstacle_cloud_node: on-table points, robot + cans removed, base frame)
      --pnp_core.scene_model: cluster + fit box-->  measured boxes (size, position, yaw), smoothed over frames
    /scene/known_obstacles (the sim's box list = ground truth; on a real robot: a measured map, or nothing)

    -> /scene/model_markers  (RViz: blue = measured boxes with their size, white wireframe = truth, error text)
    -> /scene/model          (std_msgs/String JSON: measured boxes + per-box errors; tools/scene_check.py prints it)

Started by bringup.launch.py with sensor_view:=true (and obstacle_source:=perceived, which provides the points).
"""

import json

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from pnp_core.catalog import load_robot, load_yaml
from pnp_core.scene_model import cluster_objects, compare_boxes, fit_box


def _quat_z(yaw):
    return 0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))


class SceneModelNode(Node):
    def __init__(self):
        super().__init__("pnp_scene_model")
        self.declare_parameter("smoothing", 0.3)       # weight of the newest frame (exponential moving average)
        self.declare_parameter("scene", "config/scenes/basic.yaml")
        t = load_yaml(self.get_parameter("scene").value)["table"]
        self.table_z = t["center"][2] + t["size"][2] / 2
        self.base = load_robot()["frames"]["base"]
        self.tracks = []        # smoothed measured boxes
        self.truth = []
        self.create_subscription(PointCloud2, "/perception/object_points", self._points, 5)
        self.create_subscription(String, "/scene/known_obstacles", self._truth, 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/scene/model_markers", 5)
        self.pub_js = self.create_publisher(String, "/scene/model", 5)
        self.get_logger().info("scene model ready -> /scene/model_markers, /scene/model")

    def _truth(self, msg):
        d = json.loads(msg.data)
        # the sim's layout + any shape added by hand in Isaac (sim/sim_services.py hand_boxes): both are truth
        self.truth = d.get("boxes", []) + d.get("hand_boxes", [])

    def _points(self, msg):
        p = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z")).astype(np.float64)
        boxes = [fit_box(c, self.table_z) for c in cluster_objects(p)]
        self.tracks = self._smooth(boxes)
        report = compare_boxes(self.tracks, self.truth) if self.truth else []
        self.pub_js.publish(String(data=json.dumps({"measured": self.tracks, "vs_truth": report})))
        self.pub_mk.publish(self._markers(msg.header, report))

    def _smooth(self, boxes):
        """Average each box with the previous frame's box at the same place (< 5 cm): steadier numbers in RViz."""
        a = self.get_parameter("smoothing").value
        out = []
        for b in boxes:
            prev = min(self.tracks, key=lambda t: np.hypot(t["center"][0] - b["center"][0], t["center"][1] - b["center"][1]),
                       default=None)
            if prev and np.hypot(prev["center"][0] - b["center"][0], prev["center"][1] - b["center"][1]) < 0.05:
                dyaw = (b["yaw"] - prev["yaw"] + np.pi / 2) % np.pi - np.pi / 2
                b = {**b, "center": list(a * np.array(b["center"]) + (1 - a) * np.array(prev["center"])),
                     "size": list(a * np.array(b["size"]) + (1 - a) * np.array(prev["size"])),
                     "yaw": prev["yaw"] + a * dyaw}
            out.append(b)
        return out

    def _markers(self, header, report):
        ms = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ms.markers.append(clear)
        header.frame_id = self.base
        for k, b in enumerate(self.tracks):
            cube = Marker(header=header, ns="measured", id=k, type=Marker.CUBE)
            cube.pose.position.x, cube.pose.position.y, cube.pose.position.z = b["center"]
            (cube.pose.orientation.x, cube.pose.orientation.y, cube.pose.orientation.z,
             cube.pose.orientation.w) = _quat_z(b["yaw"])
            cube.scale.x, cube.scale.y, cube.scale.z = [max(v, 0.005) for v in b["size"]]
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = 0.2, 0.5, 1.0, 0.45
            txt = Marker(header=header, ns="measured_size", id=k, type=Marker.TEXT_VIEW_FACING)
            txt.pose.position.x, txt.pose.position.y = b["center"][:2]
            txt.pose.position.z = b["center"][2] + b["size"][2] / 2 + 0.05
            txt.scale.z = 0.025
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = "measured {:.1f} x {:.1f} x {:.1f} cm".format(*[v * 100 for v in b["size"]])
            for r in report:
                if r.get("measured_index") == k:
                    e = r["size_err_mm"]
                    txt.text += f"\nvs true '{r['id']}': size err {e[0]:+.0f}/{e[1]:+.0f}/{e[2]:+.0f} mm, " \
                                f"pos err {r['center_err_mm']:.0f} mm, yaw err {r['yaw_err_deg']:+.0f} deg"
            ms.markers += [cube, txt]
        for k, t in enumerate(self.truth):
            ms.markers.append(self._wireframe(header, k, t))
        return ms

    @staticmethod
    def _wireframe(header, k, box):
        """White edges of a TRUE box (the sim's ground truth)."""
        m = Marker(header=header, ns="truth", id=k, type=Marker.LINE_LIST)
        m.scale.x = 0.003
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
        hx, hy, hz = np.asarray(box["size"]) / 2
        c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
        corners = [np.array(box["center"]) + np.array([c * x - s * y, s * x + c * y, z])
                   for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count("1") == 1:          # corners that differ in exactly one axis = an edge
                    m.points += [Point(x=float(corners[i][0]), y=float(corners[i][1]), z=float(corners[i][2])),
                                 Point(x=float(corners[j][0]), y=float(corners[j][1]), z=float(corners[j][2]))]
        return m


def main():
    rclpy.init()
    node = SceneModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
