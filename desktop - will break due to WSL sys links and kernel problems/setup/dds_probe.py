"""Minimal ROS 2 connectivity probe: is the DDS link between two machines/containers working?

Run `pong` on one side and `ping` on the other. `ping` publishes /pnp_probe/ping, `pong`
echoes it back on /pnp_probe/pong, and `ping` reports the round-trip time. Success means
discovery AND data flow work in both directions.

  Linux / container:  python3 setup/dds_probe.py pong
  Windows (Isaac's bundled ROS 2 libs, no simulator needed):
      . .\\setup\\windows_env.ps1
      $env:PYTHONPATH = "C:\\isaac-sim\\exts\\isaacsim.ros2.bridge\\jazzy\\rclpy"
      C:\\isaac-sim\\python.bat setup\\dds_probe.py ping
"""

import sys
import time

import rclpy
from std_msgs.msg import String


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "ping"
    timeout_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    rclpy.init()
    node = rclpy.create_node(f"pnp_probe_{mode}")

    if mode == "list":   # e.g. `dds_probe.py list 8 /pnp_stress` -> topics visible from this side
        prefix = sys.argv[3] if len(sys.argv) > 3 else "/"
        end = time.time() + timeout_s
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
        names = sorted(n for n, _ in node.get_topic_names_and_types() if n.startswith(prefix))
        print(f"[probe] {len(names)} topics under {prefix}: {' '.join(names)}", flush=True)
        return 0

    if mode == "pong":
        pub = node.create_publisher(String, "/pnp_probe/pong", 10)
        node.create_subscription(String, "/pnp_probe/ping", lambda m: pub.publish(m), 10)
        print("[probe] pong: echoing /pnp_probe/ping -> /pnp_probe/pong (Ctrl+C to stop)", flush=True)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        return 0

    rtts = []
    pub = node.create_publisher(String, "/pnp_probe/ping", 10)
    node.create_subscription(
        String, "/pnp_probe/pong", lambda m: rtts.append(time.perf_counter() - float(m.data)), 10
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline and len(rtts) < 5:
        pub.publish(String(data=repr(time.perf_counter())))
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()
    if rtts:
        print(f"[probe] OK: {len(rtts)} round trips, median RTT {sorted(rtts)[len(rtts) // 2] * 1e3:.1f} ms", flush=True)
        return 0
    print(f"[probe] FAIL: no pong within {timeout_s:.0f} s (discovery or data path blocked)", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
