"""pnp_core.servo_error: the sim's model of real servo joints (offsets, dead zone, gravity flex, encoder)."""
import numpy as np
import pytest

from pnp_core import robot_description as rdesc
from pnp_core.catalog import load_robot, load_yaml, repo_root
from pnp_core.servo_error import GravityModel, ServoErrorModel

CFG = load_robot("so101")
JOINTS = CFG["arm"]["joints"]
HAVE = (repo_root() / CFG["description"]["urdf"]).exists()


def model(**kw):
    base = dict(offset={j: 0.0 for j in JOINTS}, deadband={j: 0.0 for j in JOINTS}, compliance={j: 0.0 for j in JOINTS})
    base.update(kw)
    return ServoErrorModel(**base)


def test_offsets_are_invisible_to_the_encoders():
    """The arm goes somewhere else, but reports exactly what was commanded: only an outside measurement sees it."""
    m = model(offset={j: np.radians(1.0) for j in JOINTS})
    cmd = {j: 0.3 for j in JOINTS}
    true = m.command(cmd)
    assert all(abs(true[j] - 0.3 - np.radians(1.0)) < 1e-12 for j in JOINTS)
    assert all(abs(v - 0.3) < 1e-12 for v in m.report(true).values())


def test_dead_zone_is_a_hysteresis():
    j = JOINTS[0]
    m = model(deadband={jj: np.radians(1.0) for jj in JOINTS})
    m.command({j: 0.0})
    up = m.command({j: 0.1})[j]                    # approached from below: stops half the band short
    assert abs(up - (0.1 - np.radians(0.5))) < 1e-12
    assert m.command({j: 0.1 - np.radians(0.3)})[j] == up        # a small step BACK stays inside the band
    assert abs(m.command({j: 0.1 + np.radians(0.3)})[j] - (up + np.radians(0.3))) < 1e-12   # onward: follows
    down = m.command({j: 0.0})[j]                  # approached from above: stops short on the other side
    assert abs(down - np.radians(0.5)) < 1e-12


def test_other_joints_pass_through_and_encoder_is_quantized():
    m = model(quantum=np.radians(0.088))
    out = m.report({"gripper": 0.123456, JOINTS[0]: 0.123456})
    assert out["gripper"] == 0.123456
    assert abs(out[JOINTS[0]] / np.radians(0.088) - round(out[JOINTS[0]] / np.radians(0.088))) < 1e-9


def test_profile_offsets_are_reproducible():
    prof = load_yaml("config/servo_error.yaml")["profiles"]["typical"]
    a, b = ServoErrorModel.from_profile(prof, JOINTS), ServoErrorModel.from_profile(prof, JOINTS)
    assert a.offset == b.offset and max(abs(v) for v in a.offset.values()) <= np.radians(prof["offset_deg"]) + 1e-12
    assert ServoErrorModel.from_profile(prof, JOINTS, seed=7).offset != a.offset


@pytest.mark.skipif(not HAVE, reason="SO-101 description not present")
def test_gravity_loads_the_shoulder_when_the_arm_reaches_out():
    g = GravityModel(rdesc.urdf(CFG, repo_root(), "mock"), CFG["frames"]["base"], JOINTS)
    home = g.torques(dict(zip(JOINTS, CFG["arm"]["home"])))                # the "candle": arm straight up
    reach = g.torques(dict(zip(JOINTS, [0.0, 0.0, 0.0, 0.0, 0.0])))        # zero pose: arm straight forward
    assert 0.4 < abs(reach["shoulder_lift"]) < 0.7                        # 0.39 kg ~14 cm out: ~0.54 N m
    assert abs(reach["elbow_flex"]) > 0.3
    assert abs(home["elbow_flex"]) < 0.05                                 # the candle unloads the elbow and wrist,
    assert 0.3 < abs(home["shoulder_lift"]) < 0.5                         # but the arm leans back over the shoulder
    assert abs(reach["shoulder_pan"]) < 1e-4                              # a vertical axis carries no gravity torque


@pytest.mark.skipif(not HAVE, reason="SO-101 description not present")
def test_joint_offsets_folded_into_the_urdf_equal_shifted_angles():
    """robot_description.urdf(offsets=...) = the nominal arm at (encoder angle + offset): what the fit assumes."""
    from pnp_core.urdf_kin import Chain
    off = dict(zip(JOINTS, np.radians([0.5, 0.9, -0.7, 0.9, -0.4])))
    nominal = Chain(rdesc.urdf(CFG, repo_root(), "mock", offsets={}), "base_link", CFG["frames"]["tcp"])
    shifted = Chain(rdesc.urdf(CFG, repo_root(), "mock", offsets=off), "base_link", CFG["frames"]["tcp"])
    for q in ([0.0] * 5, [0.3, -1.0, 0.8, 0.5, -1.2], CFG["arm"]["home"]):
        q = np.array(q, float)
        assert np.allclose(shifted.fk(q), nominal.fk(q + np.array([off[j] for j in nominal.names])), atol=1e-9)
