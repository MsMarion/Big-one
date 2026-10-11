"""Where are MoveIt's occupied voxels? Decodes the Octomap of the planning scene and prints clusters (debugging
"MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE" / "START_STATE_IN_COLLISION" with <octomap>).

    python3 /repo/tools/octomap_dump.py [--min-z 0.02] [--watch 10]

The full (non-binary) Octomap serialization, depth first: per node a float32 log-odds value, then one byte whose
bit i says child i exists. Leaves at depth 16 are voxels of `resolution`; the key of the tree center is 2^15.
"""

import argparse
import struct
import sys
import time

import numpy as np
import rclpy
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rclpy.node import Node

sys.path.insert(0, "/repo")
from pnp_core.map_monitor import clusters, summarize  # noqa: E402

TREE_DEPTH = 16


def decode(data: bytes, res: float, threshold_logodds: float = 0.0):
    """-> (N,3) centers of occupied leaf voxels."""
    out, pos = [], [0]

    def node(depth, key):
        v, = struct.unpack_from("<f", data, pos[0])
        children = data[pos[0] + 4]
        pos[0] += 5
        if children == 0:
            if v > threshold_logodds:
                size = 1 << (TREE_DEPTH - depth)                      # leaf may be a pruned bigger node
                out.append((key, size))
            return
        half = 1 << (TREE_DEPTH - depth - 1)
        for i in range(8):
            if children & (1 << i):
                node(depth + 1, (key[0] + (half if i & 1 else 0), key[1] + (half if i & 2 else 0),
                                 key[2] + (half if i & 4 else 0)))
    if data:
        node(0, (0, 0, 0))
    pts = []
    for (kx, ky, kz), size in out:
        for dx in range(size):
            for dy in range(size):
                for dz in range(size):
                    pts.append(((kx + dx - 32768 + 0.5) * res, (ky + dy - 32768 + 0.5) * res,
                                (kz + dz - 32768 + 0.5) * res))
    return np.array(pts).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-z", type=float, default=0.02, help="ignore voxels below (the table surface layer)")
    ap.add_argument("--watch", type=float, default=0.0, help="repeat every second for this many seconds")
    a = ap.parse_args()
    rclpy.init()
    n = Node("pnp_octomap_dump")
    cli = n.create_client(GetPlanningScene, "/get_planning_scene")
    cli.wait_for_service(timeout_sec=5)
    end = time.time() + a.watch
    while True:
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.OCTOMAP
        f = cli.call_async(req)
        rclpy.spin_until_future_complete(n, f, timeout_sec=5)
        om = f.result().scene.world.octomap.octomap
        if om.binary:
            print("binary octomap: not supported by this decoder")
            return 1
        pts = decode(bytes(om.data), om.resolution)
        low = pts[pts[:, 2] < a.min_z] if len(pts) else pts
        high = pts[pts[:, 2] >= a.min_z] if len(pts) else pts
        cs = summarize(clusters({tuple(k) for k in np.floor(high / om.resolution).astype(int)}, 1), om.resolution)
        print(f"{time.strftime('%X')} occupied voxels: {len(pts)} ({len(low)} below {a.min_z} m); "
              f"above: {len(high)} in {len(cs)} clusters " + "; ".join(f"{c['voxels']}@{c['center']}" for c in cs[:6]),
              flush=True)
        if time.time() >= end:
            break
        time.sleep(1.0)
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
