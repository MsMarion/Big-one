"""The outer integral loop (pnp_core/servo_hold.py) on a simulated loaded P-servo."""
from pnp_core.servo_hold import HoldIntegrator


def servo(sent: float, load: float, kp: float = 1.0) -> float:
    """A P servo under a constant load: it settles short of what it was sent by load / kp."""
    return sent - load / kp


def run(hi, goal, load, steps=200, dt=0.02, moving=False, dq=0.0):
    q = goal
    for _ in range(steps):
        off = hi.update({"j": goal}, {"j": q}, {"j": dq}, {"j": moving}, dt)
        q = servo(goal + off["j"], load)
    return q, off["j"]


def test_a_sagging_joint_reaches_its_goal():
    hi = HoldIntegrator(["j"])
    q, off = run(hi, 0.5, load=0.03)                          # sags 0.03 rad (1.7 deg) without the loop
    assert abs(q - 0.5) < 1e-3 and abs(off - 0.03) < 1e-3


def test_nothing_happens_while_moving_or_blocked_and_it_is_clamped():
    hi = HoldIntegrator(["j"])
    assert run(hi, 0.5, load=0.03, moving=True)[1] == 0.0     # a motion is never fought
    assert run(hi, 0.5, load=0.03, dq=0.2)[1] == 0.0          # nor a joint still moving
    assert run(hi, 0.5, load=0.3)[1] == 0.0                   # 17 deg off: pushing against something -> hands off
    hi2 = HoldIntegrator(["j"], max_rad=0.02)
    q, off = run(hi2, 0.5, load=0.05, steps=2000)             # more sag than the cap allows: capped, not wound up
    assert off == 0.02 and abs(q - 0.47) < 1e-6


def test_disabled_and_reset():
    hi = HoldIntegrator(["j"], ki=0.0)
    assert run(hi, 0.5, load=0.03)[1] == 0.0
    hi = HoldIntegrator(["j"])
    run(hi, 0.5, load=0.03)
    hi.reset()
    assert hi.off["j"] == 0.0
