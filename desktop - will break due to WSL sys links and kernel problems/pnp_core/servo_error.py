"""How a real hobby servo arm (SO-101, STS3215) differs from its encoders: a joint error model for the sim.

The sim's joints go exactly where they are told and report exactly where they are. A real STS3215 arm doesn't:
  offset      the calibration (homing) zero is off by a little per joint          INVISIBLE to the encoder
  deadband    the servo stops anywhere within its dead zone of the target, on the side it came from
              (a hysteresis: the output moves only once the command leaves the band)      VISIBLE (encoder on the shaft)
  flex        the horn and the printed links bend under gravity: deflection = compliance x gravity torque
                                                                                    INVISIBLE (after the encoder)
  encoder     noise + 12-bit quantization (4096 steps / turn = 0.088 deg)                  VISIBLE
pnp_manipulation/servo_error_node.py puts this between ros2_control and Isaac (bringup servo_error:=<profile>,
config/servo_error.yaml): commands are turned into where the joint REALLY goes, Isaac's true joint angles into what the
encoders would REPORT. Tools that measure the arm from outside (the cameras: tools/arm_accuracy_check.py) then see
the same kind of error as on the real arm. docs/learning/17_so101.md "Servo errors".
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .urdf_kin import Chain

G = np.array([0.0, 0.0, -9.81])


class GravityModel:
    """Gravity torque on each joint of a URDF arm (static: no motion), from the links' masses and centers of mass."""

    def __init__(self, urdf_xml: str, base: str, joints: list):
        root = ET.fromstring(urdf_xml)
        self.joints = list(joints)
        self.links = []                                   # (chain base -> link, mass, com in the link frame)
        for link in root.findall("link"):
            ine = link.find("inertial")
            if ine is None or link.get("name") == base:
                continue
            m = float(ine.find("mass").get("value"))
            if m < 1e-6:
                continue
            o = ine.find("origin")
            com = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
            try:
                ch = Chain(urdf_xml, base, link.get("name"))
            except ValueError:
                continue
            self.links.append((ch, m, com))
        self.joint_chains = {j: self._chain_to_joint(urdf_xml, base, root, j) for j in self.joints}

    @staticmethod
    def _chain_to_joint(urdf_xml, base, root, joint):
        child = root.find(f"joint[@name='{joint}']").find("child").get("link")
        return Chain(urdf_xml, base, child)

    def torques(self, q: dict) -> dict:
        """{joint: gravity torque about its axis [N m]} at joint values q (missing joints: 0)."""
        out = {}
        for j in self.joints:
            ch = self.joint_chains[j]
            T = ch.fk([q.get(n, 0.0) for n in ch.names])      # the joint's child frame (its axis is fixed in it)
            axis = T[:3, :3] @ ch.joints[-1].axis
            p = T[:3, 3]
            tau = 0.0
            for lch, m, com in self.links:
                if j not in lch.names:                         # only links the joint carries
                    continue
                Tl = lch.fk([q.get(n, 0.0) for n in lch.names])
                c = Tl[:3, :3] @ com + Tl[:3, 3]
                tau += float(axis @ np.cross(c - p, m * G))
            out[j] = tau
        return out


@dataclass
class ServoErrorModel:
    offset: dict                       # joint -> rad (true = encoder + offset)
    deadband: dict                     # joint -> rad, full width of the dead zone
    compliance: dict                   # joint -> rad per N m of gravity torque
    noise: float = 0.0                 # rad, encoder noise sigma
    quantum: float = 0.0               # rad, encoder resolution
    gravity: GravityModel | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _out: dict = field(default_factory=dict)          # where each servo's output stopped (the hysteresis state)
    _flex_last: dict = field(default_factory=dict)    # flex at the last reported state (reused by command())

    @classmethod
    def from_profile(cls, prof: dict, joints: list, gravity: GravityModel | None = None, seed: int | None = None):
        """config/servo_error.yaml profile -> model. Offsets are drawn per joint from +- offset_deg (seeded: the
        same arm every run), unless the profile lists them per joint (offsets_deg: {joint: deg})."""
        rng = np.random.default_rng(prof.get("seed", 0) if seed is None else seed)
        r = np.radians
        per = prof.get("offsets_deg", {})
        if prof.get("offsets_from"):      # the twin: the real arm's fitted joint offsets (tools/fit_joint_offsets.py)
            import yaml
            from .catalog import repo_root
            per = (yaml.safe_load((repo_root() / prof["offsets_from"]).read_text()) or {}).get("offsets_deg") or {}
        off = {j: r(per[j]) if j in per else r(rng.uniform(-1, 1) * prof.get("offset_deg", 0.0)) for j in joints}
        return cls(offset=off, deadband={j: r(prof.get("deadband_deg", 0.0)) for j in joints},
                   compliance={j: r(prof.get("compliance_deg_per_nm", 0.0)) for j in joints},
                   noise=r(prof.get("noise_deg", 0.0)), quantum=r(prof.get("quantum_deg", 0.0)), gravity=gravity,
                   rng=rng)

    def _flex(self, q: dict) -> dict:
        if self.gravity is None:
            return {}
        tau = self.gravity.torques(q)
        return {j: self.compliance.get(j, 0.0) * t for j, t in tau.items()}

    def command(self, cmd: dict, q_true: dict | None = None) -> dict:
        """Commanded encoder angles -> the joint angles the arm REALLY goes to (the sim's targets). The gravity flex
        is evaluated at q_true, or (None) taken from the last report() (the arm's current state)."""
        flex = self._flex(q_true) if q_true is not None else self._flex_last
        out = {}
        for j, c in cmd.items():
            if j not in self.offset:
                out[j] = c
                continue
            half = self.deadband[j] / 2
            o = self._out.get(j, c)
            o = min(max(o, c - half), c + half)               # the output moves only when the command leaves the band
            self._out[j] = o
            out[j] = o + self.offset[j] + flex.get(j, 0.0)
        return out

    def report(self, q_true: dict) -> dict:
        """True joint angles (Isaac) -> what the encoders report."""
        flex = self._flex_last = self._flex(q_true)
        out = {}
        for j, q in q_true.items():
            if j not in self.offset:
                out[j] = q
                continue
            e = q - self.offset[j] - flex.get(j, 0.0)
            if self.noise:
                e += self.rng.normal(0.0, self.noise)
            if self.quantum:
                e = np.round(e / self.quantum) * self.quantum
            out[j] = float(e)
        return out

    def truth(self, q_true: dict) -> dict:
        """For checks: each joint's current invisible error (true - what the encoder implies) [deg]."""
        flex = self._flex(q_true)
        return {j: {"offset_deg": round(float(np.degrees(self.offset[j])), 3),
                    "flex_deg": round(float(np.degrees(flex.get(j, 0.0))), 3),
                    "deadband_deg": round(float(np.degrees(self.deadband[j])), 3)} for j in self.offset}
