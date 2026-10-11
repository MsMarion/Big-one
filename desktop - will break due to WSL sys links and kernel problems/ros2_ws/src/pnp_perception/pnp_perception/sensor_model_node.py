"""Sensor model (sim only): perfect Isaac images in, realistic camera images out. docs/learning/15_sim_realism.md

    started by bringup.launch.py when sim:=true;  profile live:  ros2 param set /pnp_sensor_model profile realistic
                                                  (tools/set_reality.py also moves the sim cameras)

Isaac publishes every camera under /sim_raw/... (config/interfaces.yaml sim_raw_prefix). This node publishes the
STANDARD topics the real drivers publish (/camera/camera/..., /wrist_camera/camera/...), with the depth and color
noise of the active profile in config/sim_reality.yaml (pnp_core/sensor_noise.py). Profile 'ideal' passes the
images through unchanged. Every consumer (detector, pose, obstacle map) is unaware of it, as with a real camera.

Depth goes out as 16UC1 millimeters (what realsense2_camera publishes) when noise is on: the 1 mm quantization is
real too.
"""

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CameraInfo, Image

from pnp_core.catalog import load_fixed_camera, load_robot, load_yaml
from pnp_core.sensor_noise import noisy_color, noisy_depth
from pnp_core.wrist import wrist_camera


def _to_np(m: Image) -> np.ndarray:
    dt = {"rgb8": np.uint8, "bgr8": np.uint8, "rgba8": np.uint8, "32FC1": np.float32, "16UC1": np.uint16}[m.encoding]
    a = np.frombuffer(m.data, dt).reshape(m.height, m.width, -1).squeeze()
    return a.astype(np.float32) / 1000.0 if m.encoding == "16UC1" else a


class SensorModel(Node):
    def __init__(self):
        super().__init__("pnp_sensor_model")
        reality = load_yaml("config/sim_reality.yaml")
        self.profiles = reality["profiles"]
        self.declare_parameter("profile", reality.get("active", "ideal"))
        self.declare_parameter("seed", 0)
        self.seed = int(self.get_parameter("seed").value)
        self.add_on_set_parameters_callback(self._on_params)
        raw = load_yaml("config/interfaces.yaml").get("sim_raw_prefix", "/sim_raw")
        t = load_yaml("config/interfaces.yaml")["topics"]
        cams = [("d435i", t["color_image"]["name"], t["aligned_depth"]["name"], t["color_info"]["name"])]
        w = wrist_camera(load_robot())
        if w:
            cams.append((w["name"], w["topics"]["color_image"], w["topics"]["aligned_depth"], w["topics"]["color_info"]))
        # the intrinsics the sim RENDERS with (sim/scene.py set_pinhole: fx, fy and an off-centre principal point as
        # aperture offsets). Isaac's camera-info helper ignores the offsets and fy (it published cx 320 / cy 240 / fy = fx
        # for the plate twin's D435i rendered at cy 256: every tag corner 16 px off, locate tilted 1.2 deg, 2026-10-11),
        # so the camera_info is rebuilt from the same numbers the camera was set up with.
        self.intr = {"d435i": load_fixed_camera()}
        if w:
            self.intr[w["name"]] = w
        q = qos_profile_sensor_data
        for model, color, depth, info in cams:
            pub_c, pub_d, pub_i = (self.create_publisher(Image, color, q), self.create_publisher(Image, depth, q),
                                   self.create_publisher(CameraInfo, info, q))
            # one callback group per stream: the streams are processed in parallel (numpy releases the GIL), each
            # stream stays in order. Single-threaded, the 4 image streams lost ~15 % of their frames.
            g = [MutuallyExclusiveCallbackGroup() for _ in range(3)]
            self.create_subscription(Image, raw + color, lambda m, p=pub_c, k=model: self._color(m, p, k), q,
                                     callback_group=g[0])
            self.create_subscription(Image, raw + depth, lambda m, p=pub_d, k=model: self._depth(m, p, k), q,
                                     callback_group=g[1])
            self.create_subscription(CameraInfo, raw + info, lambda m, p=pub_i, k=model: p.publish(self._info(m, k)), q,
                                     callback_group=g[2])
        self.get_logger().info(f"sensor model: {[c[0] for c in cams]}, profile '{self.get_parameter('profile').value}' "
                               f"({raw}/... -> standard topics)")

    def _info(self, m: CameraInfo, model: str) -> CameraInfo:
        """Isaac's camera_info with K / P from the intrinsics the camera renders with (see __init__)."""
        c = self.intr.get(model) or {}
        if "fx" not in c:
            return m
        fx, fy = float(c["fx"]), float(c.get("fy", c["fx"]))
        cx, cy = float(c.get("cx", m.width / 2)), float(c.get("cy", m.height / 2))
        m.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        m.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return m

    def _on_params(self, params):
        self.profiles = load_yaml("config/sim_reality.yaml")["profiles"]      # re-read: edits apply without a restart
        for p in params:
            if p.name == "profile" and p.value not in self.profiles:
                return SetParametersResult(successful=False, reason=f"unknown profile; known: {sorted(self.profiles)}")
        for p in params:
            if p.name == "profile":
                self.get_logger().info(f"sensor model profile -> {p.value}")
        return SetParametersResult(successful=True)

    def _rng(self):
        """One generator per thread (numpy generators are not thread-safe)."""
        import threading
        loc = self.__dict__.setdefault("_tls", threading.local())
        if not hasattr(loc, "rng"):
            loc.rng = np.random.default_rng([self.seed, threading.get_ident() & 0xFFFFFFFF])
        return loc.rng

    def _params(self, model):
        return self.profiles[self.get_parameter("profile").value].get(model, {})

    def _color(self, m: Image, pub, model):
        p = self._params(model)
        if p.get("color_sigma", 0) <= 0:
            pub.publish(m)
            return
        out = noisy_color(_to_np(m), p, self._rng())
        m.data = out.tobytes()
        pub.publish(m)

    def _depth(self, m: Image, pub, model):
        p = self._params(model)
        if p.get("k", 0) <= 0 and not p.get("holes"):
            pub.publish(m)                                   # ideal: untouched (Isaac's 32FC1 meters)
            return
        z = noisy_depth(_to_np(m), p, self._rng())
        mm = np.clip(np.round(z * 1000.0), 0, 65535).astype(np.uint16)   # like realsense2_camera: 16UC1 mm
        out = Image(header=m.header, height=m.height, width=m.width, encoding="16UC1", is_bigendian=0,
                    step=m.width * 2, data=mm.tobytes())
        pub.publish(out)


def main():
    rclpy.init()
    n = SensorModel()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(n)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
