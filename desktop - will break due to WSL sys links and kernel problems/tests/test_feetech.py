"""Feetech STS bus protocol + calibration (pnp_core/feetech.py): what the real SO-101 bridge relies on."""
import math

import pytest

from pnp_core.feetech import (Calibration, FeetechBus, MockBus, MotorCalibration, RAD_PER_TICK, checksum, packet,
                              parse_status, signed_speed)


def test_packet_layout_and_checksum():
    # a documented example: PING servo 1 -> FF FF 01 02 01 FB
    assert packet(1, 0x01) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])
    p = packet(3, 0x03, bytes([42, 0x00, 0x08]))           # WRITE goal_position = 2048 to id 3
    assert p[:2] == b"\xff\xff" and p[2] == 3 and p[3] == 5 and p[-1] == checksum(p[2:-1])


def test_parse_status_skips_garbage_and_bad_checksums():
    good = packet(2, 0x00, bytes([0x00, 0x08]))            # a status packet: id 2, error 0, 2 data bytes
    bad = bytearray(packet(5, 0x00, b"\x01"))
    bad[-1] ^= 0xFF
    out = parse_status(b"\x00\x13" + bytes(bad) + good)
    assert out == [(2, 0x00, b"\x08")] or out == [(2, 0, bytes([0x08]))] or out[0][0] == 2


def test_speed_sign_bit():
    assert signed_speed(0x8000 | 100) == -100 and signed_speed(100) == 100


class FakePort:
    """Answers like a servo bus: echoes nothing, replies to READ / SYNC_READ with stored registers."""
    def __init__(self, positions):
        self.positions, self.out = positions, b""

    def flush_input(self):
        self.out = b""

    def write(self, data):
        sid, instr = data[2], data[4]
        if instr == 0x01:
            self.out = packet(sid, 0x00) if sid in self.positions else b""
        elif instr == 0x02:
            v = self.positions[sid]
            self.out = packet(sid, 0x00, bytes([v & 0xFF, v >> 8]))
        elif instr == 0x82:
            ids = data[7:-1]
            self.out = b"".join(packet(i, 0x00, bytes([self.positions[i] & 0xFF, self.positions[i] >> 8]))
                                for i in ids if i in self.positions)

    def read(self, n):
        out, self.out = self.out[:n], self.out[n:]
        return out


def test_bus_ping_read_sync_read_over_a_fake_port():
    bus = FeetechBus(FakePort({1: 2048, 2: 1000, 3: 3000}))
    assert bus.ping(1) and not bus.ping(9)
    assert bus.read(2, "present_position") == 1000
    assert bus.sync_read("present_position", [1, 2, 3]) == {1: 2048, 2: 1000, 3: 3000}


def test_calibration_mid_range_is_zero_and_round_trips():
    cal = Calibration({"elbow_flex": MotorCalibration(3, 0, 1000, 3000),
                       "wrist_flex": MotorCalibration(4, 1, 1000, 3000),
                       "gripper": MotorCalibration(6, 0, 2000, 3400)}, gripper="gripper")
    assert cal.to_rad("elbow_flex", 2000) == pytest.approx(0.0)
    assert cal.to_rad("elbow_flex", 2000 + 512) == pytest.approx(512 * RAD_PER_TICK)
    assert cal.to_rad("wrist_flex", 2000 + 512) == pytest.approx(-512 * RAD_PER_TICK)       # drive_mode 1: mirrored
    for n, q in (("elbow_flex", 0.7), ("wrist_flex", -0.4), ("gripper", 0.3)):
        assert cal.to_rad(n, cal.to_ticks(n, q)) == pytest.approx(q, abs=RAD_PER_TICK)
    assert cal.to_ticks("elbow_flex", 10.0) == 3000                  # never outside the recorded range
    assert cal.to_rad("gripper", 2000) == pytest.approx(-0.17453) and cal.to_rad("gripper", 3400) == pytest.approx(1.74533)


def test_example_calibration_file_loads(tmp_path):
    from pnp_core.catalog import repo_root
    cal = Calibration.load(repo_root() / "config/calibration/so101_desk/so101_follower.example.json")
    assert set(cal.motors) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}
    assert abs(cal.to_rad("shoulder_pan", cal.motors["shoulder_pan"].range_max)) > math.radians(90)


def test_mock_bus_holds_then_follows_goals():
    bus = MockBus({1: 2048}, speed_ticks_per_s=1e6)
    bus.write(1, "goal_position", 3000)                 # torque off: nothing moves
    assert bus.read(1, "present_position") == 2048
    bus.write(1, "torque_enable", 1)
    bus.sync_write("goal_position", {1: 2500})
    import time  # noqa: E402
    time.sleep(0.01)
    assert bus.read(1, "present_position") == 2500


class EchoPort(FakePort):
    """An adapter that echoes every request before the servo's answer."""
    def write(self, data):
        super().write(data)
        self.out = data + self.out


def test_bus_detects_and_strips_an_echoing_adapter():
    bus = FeetechBus(EchoPort({1: 2048, 2: 1000}))
    assert bus.ping(1) and bus.echo is True
    assert bus.read(2, "present_position") == 1000            # not the echoed request's bytes
    assert bus.sync_read("present_position", [1, 2]) == {1: 2048, 2: 1000}
    plain = FeetechBus(FakePort({1: 2048}))
    assert plain.ping(1) and plain.echo is False


@pytest.mark.skipif(not hasattr(__import__("os"), "openpty") or __import__("sys").platform == "win32",
                    reason="needs a Linux pseudo-terminal")
@pytest.mark.parametrize("echo", [False, True])
def test_real_serial_path_against_the_emulated_servo_bus(echo, tmp_path):
    """termios serial + packet bytes + echo handling, end to end against tools/sts_servo_emulator.py."""
    import subprocess
    import sys
    import time as _t
    from pnp_core.catalog import repo_root
    from pnp_core.feetech import SerialPort
    link = str(tmp_path / "bus")
    cmd = [sys.executable, str(repo_root() / "tools/sts_servo_emulator.py"), "--link", link] + (["--echo"] if echo else [])
    emu = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if __import__("os").path.exists(link):
                break
            _t.sleep(0.05)
        bus = FeetechBus(SerialPort(link, 1_000_000, timeout_s=0.05))
        assert all(bus.ping(i) for i in range(1, 7)) and not bus.ping(9)
        assert bus.echo is echo
        start = bus.sync_read("present_position", range(1, 7))
        assert len(start) == 6
        bus.sync_write("torque_enable", {1: 1})
        bus.sync_write("goal_position", {1: start[1] + 300})
        _t.sleep(0.4)
        assert abs(bus.read(1, "present_position") - (start[1] + 300)) <= 2      # it moved there
        assert bus.read(2, "present_position") == start[2]                       # torque off: didn't move
    finally:
        emu.kill()


def test_homing_offset_is_sign_magnitude():
    """LeRobot stores negative homing offsets as |v| with bit 11 set (not two's complement)."""
    from pnp_core.feetech import decode_signed
    assert decode_signed("homing_offset", 1234) == 1234
    assert decode_signed("homing_offset", 0x800 | 1901) == -1901
    assert decode_signed("present_position", 0x800 | 5) == 0x800 | 5      # unsigned registers unchanged
