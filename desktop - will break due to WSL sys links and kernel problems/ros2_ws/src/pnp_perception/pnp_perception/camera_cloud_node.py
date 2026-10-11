"""Colored 3D point cloud from the camera: how "a point cloud" is actually made (for viewing in RViz).

    color image + aligned depth + camera_info  ->  /camera/camera/depth/color/points  (PointCloud2, x y z rgb,
                                                    camera optical frame, ~5 Hz, every 2nd pixel)

For each pixel (u, v) with depth z [m] (distance along the optical axis):
    x = (u - cx) * z / fx        y = (v - cy) * z / fy        (camera optical frame: x right, y down, z forward)
fx, fy = focal lengths in PIXELS, cx, cy = principal point, from camera_info (= config/camera_d435i.yaml in sim,
factory calibration on the real D435i). That's the whole "scale" story: depth gives meters along the ray, the
intrinsics give the ray's direction. Aligned depth means depth pixel (u, v) and color pixel (u, v) are the same
point, so each 3D point simply takes its pixel's color.

The real D435i driver can publish the same topic itself (pointcloud.enable); computing it here keeps sim and real
identical and shows the math. (depth_image_proc's point_cloud_xyzrgb_node does the same in C++.)
"""

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

from pnp_core.catalog import load_yaml
from pnp_core.perception_math import depth_to_meters

FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")] + \
         [PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)]


class CameraCloudNode(Node):
    def __init__(self):
        super().__init__("pnp_camera_cloud")
        self.declare_parameter("pixel_step", 2)
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("max_depth_m", 2.5)
        topics = load_yaml("config/interfaces.yaml")["topics"]
        self.info, self.pair, self.last = None, None, 0.0
        self.create_subscription(CameraInfo, topics["color_info"]["name"], lambda m: setattr(self, "info", m),
                                 qos_profile_sensor_data)
        color = message_filters.Subscriber(self, Image, topics["color_image"]["name"], qos_profile=qos_profile_sensor_data)
        depth = message_filters.Subscriber(self, Image, topics["aligned_depth"]["name"], qos_profile=qos_profile_sensor_data)
        sync = message_filters.ApproximateTimeSynchronizer([color, depth], queue_size=10, slop=0.05)
        sync.registerCallback(lambda c, d: setattr(self, "pair", (c, d)))
        self.pub = self.create_publisher(PointCloud2, topics["points"]["name"], 2)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self._tick)
        self.get_logger().info(f"camera cloud -> {topics['points']['name']}")

    def _tick(self):
        if self.pair is None or self.info is None:
            return
        (cmsg, dmsg), self.pair = self.pair, None
        step = self.get_parameter("pixel_step").value
        depth = depth_to_meters(np.frombuffer(dmsg.data, dtype=np.uint16 if dmsg.encoding == "16UC1" else np.float32)
                                .reshape(dmsg.height, dmsg.width), dmsg.encoding)[::step, ::step]
        rgb = np.frombuffer(cmsg.data, np.uint8).reshape(cmsg.height, cmsg.width, -1)[::step, ::step, :3]
        if cmsg.encoding.lower().startswith("bgr"):
            rgb = rgb[..., ::-1]
        k = self.info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        vs, us = np.mgrid[0:dmsg.height:step, 0:dmsg.width:step]
        ok = (depth > 0.1) & (depth < self.get_parameter("max_depth_m").value)
        z = depth[ok]
        pts = np.zeros(int(ok.sum()), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
        pts["x"], pts["y"], pts["z"] = (us[ok] - cx) * z / fx, (vs[ok] - cy) * z / fy, z
        c = rgb[ok].astype(np.uint32)
        pts["rgb"] = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)   # PCL's packed-float color
        out = PointCloud2(header=dmsg.header, height=1, width=len(pts), fields=FIELDS, is_bigendian=False,
                          point_step=16, row_step=16 * len(pts), is_dense=True, data=pts.tobytes())
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CameraCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
