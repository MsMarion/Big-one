"""Client for the sim's request/response topics (see sim/sim_services.py for why these are topics).

    from sim_client import SimClient
    sim = SimClient(node)
    info = sim.call("randomize_target")      # -> {"id": .., "ok": True, "xy": [...], "yaw": ...}

Also a CLI:  python3 /repo/tools/sim_client.py randomize_target
             python3 /repo/tools/sim_client.py place_object '{"name": "cracker_box", "xy": [0.5, 0.0], "yaw": 0.3}'
"""

import itertools
import json
import sys
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class SimClient:
    def __init__(self, node):
        self.node = node
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = node.create_publisher(String, "/sim/request", qos)
        self.responses = {}
        node.create_subscription(String, "/sim/response", self._on_response, qos)
        self.ids = itertools.count(int(time.time() * 1000) % 1_000_000)

    def _on_response(self, msg):
        r = json.loads(msg.data)
        self.responses[r.get("id")] = r

    def wait_ready(self, timeout=15.0):
        """Wait until the sim's request subscriber is discovered (a request sent earlier would be lost)."""
        end = time.time() + timeout
        while time.time() < end:
            if self.pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return False

    def call(self, cmd, timeout=10.0, retry_every=1.0, **args):
        """Send `cmd` and wait for its response. The request is re-sent (same id) every `retry_every` s:
        right after a publisher appears, the zenoh bridge may still be building its route and drop the first
        message. The sim answers repeated ids from a cache, so a command never executes twice."""
        if not self.wait_ready():
            raise RuntimeError("sim not reachable (/sim/request has no subscriber): is run_sim.py running?")
        rid = next(self.ids)
        msg = String(data=json.dumps({"id": rid, "cmd": cmd, **args}))      # args: e.g. profile="realistic"
        end, next_send = time.time() + timeout, 0.0
        while time.time() < end:
            if time.time() >= next_send:
                self.pub.publish(msg)
                next_send = time.time() + retry_every
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if rid in self.responses:
                r = self.responses.pop(rid)
                if not r.get("ok"):
                    raise RuntimeError(f"sim error: {r.get('error')}")
                return r
        raise RuntimeError(f"no response from sim for '{cmd}' within {timeout} s")


if __name__ == "__main__":
    rclpy.init()
    n = rclpy.create_node("sim_client_cli")
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}     # e.g. place_object '{"name": "target", "xy": [0.5, 0]}'
    print(json.dumps(SimClient(n).call(sys.argv[1] if len(sys.argv) > 1 else "randomize_target", **args)))
    rclpy.shutdown()
