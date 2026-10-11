"""Where can a robot grasp? A reach map from its URDF (pnp_core/urdf_kin.py): every table point the tool frame can
reach at a given height with a given approach direction (default: straight down), within the joint limits.

    python3 /repo/tools/reach_map.py --robot so101                    # profile defaults (robots/<name>/robot.yaml)
    python3 /repo/tools/reach_map.py --urdf X.urdf --base base_link --tip tool0 --z 0.01 --extent 0.6
-> a PNG (green = reachable top-down) + the reachable area and radius range on stdout.
"""

import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, "/repo")
from pnp_core.urdf_kin import Chain, reach_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--z", type=float, default=0.01, help="grasp height above the base frame [m]")
    ap.add_argument("--extent", type=float, default=0.5, help="map +- this around the base in x and y [m]")
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--approach", default="0,0,-1")
    ap.add_argument("--out", default="/repo/data/reach_map.png")
    a = ap.parse_args()
    ch = Chain.from_file(a.urdf, a.base, a.tip)
    xs = np.arange(-a.extent, a.extent + 1e-9, a.step)
    ys = np.arange(-a.extent, a.extent + 1e-9, a.step)
    ok, _ = reach_map(ch, xs, ys, a.z, [float(v) for v in a.approach.split(",")])
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X, Y)[ok]
    print(f"{ch.names}: reachable {ok.sum() * a.step ** 2 * 1e4:.0f} cm^2 at z={a.z} m, "
          f"radius {r.min() * 100:.1f}-{r.max() * 100:.1f} cm" if ok.any() else "nothing reachable")
    px = 4
    img = np.full((len(ys) * px, len(xs) * px, 3), 255, np.uint8)
    img[np.repeat(np.repeat(ok[::-1], px, 0), px, 1)] = (80, 200, 80)
    c = (int(len(xs) / 2 * px), int(len(ys) / 2 * px))
    cv2.circle(img, c, 6, (0, 0, 200), -1)                                    # the base
    for k in range(1, int(a.extent / 0.1) + 1):                               # 10 cm rings
        cv2.circle(img, c, int(k * 0.1 / a.step * px), (200, 200, 200), 1)
    cv2.arrowedLine(img, c, (c[0] + 40, c[1]), (0, 0, 0), 2)                  # +x
    cv2.putText(img, "+x", (c[0] + 44, c[1] + 5), 0, 0.5, (0, 0, 0), 1)
    cv2.imwrite(a.out, img)
    print(f"map: {a.out} (rings every 10 cm, red = base)")


if __name__ == "__main__":
    main()
