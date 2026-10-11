"""Feetech STS servo bus (the SO-101's STS3215 motors): protocol, calibration, and a mock bus for tests.

What this teaches: hobby-grade "smart servos" sit on one half-duplex serial bus (1 Mbaud on the SO-101), each with an
ID. A host sends packets  FF FF <id> <len> <instruction> <params...> <checksum>  and the addressed servo answers.
Everything is a register in the servo's memory table: write Goal_Position (2 bytes) to move, read Present_Position
to know where it is, Torque_Enable to stiffen / relax it. SYNC_WRITE / SYNC_READ address several servos in one packet,
which is what a 50 Hz control loop needs.

Positions are raw TICKS: 4096 per turn (0.088 deg). The calibration maps ticks <-> joint radians of the robot
description. We use LeRobot's calibration file (the standard SO-101 workflow: `lerobot-calibrate`): per motor
{id, drive_mode, homing_offset, range_min, range_max}. LeRobot writes homing_offset INTO the servo, so the ticks we
read are already offset; the joint zero is the MIDDLE of [range_min, range_max] (= the URDF's "new calibration"),
drive_mode 1 = the joint turns the other way. The gripper is mapped linearly from its recorded range onto the URDF
gripper limits (closed .. open).

Pure Python, Linux serial via termios (no pyserial needed). Unit-tested with MockBus (tests/test_feetech.py).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

# STS / SMS series memory table (address, size): the registers we use
REG = {"p_coefficient": (21, 1), "homing_offset": (31, 2), "torque_enable": (40, 1), "acceleration": (41, 1), "goal_position": (42, 2), "goal_speed": (46, 2),
       "torque_limit": (48, 2), "lock": (55, 1), "present_position": (56, 2), "present_speed": (58, 2),
       "present_load": (60, 2), "present_voltage": (62, 1), "present_temperature": (63, 1), "moving": (66, 1),
       "present_current": (69, 2)}
SIGN_MAGNITUDE_BIT = {"homing_offset": 11}   # these registers store |value| + a sign bit, not two's complement


def decode_signed(reg: str, raw: int) -> int:
    """A sign-magnitude register's value (homing_offset: bit 11 = negative), as LeRobot writes it."""
    b = SIGN_MAGNITUDE_BIT.get(reg)
    if b is None or raw is None:
        return raw
    return -(raw & ((1 << b) - 1)) if raw & (1 << b) else raw


PING, READ, WRITE, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x82, 0x83
TICKS_PER_TURN = 4096
RAD_PER_TICK = 2 * math.pi / TICKS_PER_TURN


def checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    body = bytes([servo_id, len(params) + 2, instruction]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


def parse_status(data: bytes) -> list[tuple[int, int, bytes]]:
    """Status packets in a byte stream -> [(id, error, params)]. Skips garbage; drops packets with a bad checksum."""
    out, i = [], 0
    while i + 6 <= len(data):
        if data[i] != 0xFF or data[i + 1] != 0xFF:
            i += 1
            continue
        sid, length = data[i + 2], data[i + 3]
        end = i + 4 + length
        if end > len(data):
            break
        body = data[i + 2:end - 1]
        if checksum(body) == data[end - 1]:
            out.append((sid, data[i + 4], bytes(data[i + 5:end - 1])))
            i = end
        else:
            i += 1
    return out


def u16(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])            # STS: little endian


def from_u16(b: bytes) -> int:
    return b[0] | (b[1] << 8)


def signed_speed(v: int) -> int:
    """STS speeds/loads: bit 15 is the sign."""
    return -(v & 0x7FFF) if v & 0x8000 else v


# --- serial port --------------------------------------------------------------------------------------------------
class SerialPort:
    """Raw serial on Linux (termios), 8N1, any of the standard baud rates incl. 1 000 000."""

    def __init__(self, device: str, baud: int = 1_000_000, timeout_s: float = 0.02):
        import termios
        self.fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        speed = getattr(termios, f"B{baud}")
        attrs[0] = 0                                          # iflag: raw
        attrs[1] = 0                                          # oflag
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0                                          # lflag: raw, no echo
        attrs[4], attrs[5] = speed, speed
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        self.timeout = timeout_s

    def write(self, data: bytes):
        os.write(self.fd, data)

    def read(self, n: int) -> bytes:
        buf, end = b"", time.time() + self.timeout
        while len(buf) < n and time.time() < end:
            try:
                chunk = os.read(self.fd, n - len(buf))
            except BlockingIOError:
                chunk = b""
            if chunk:
                buf += chunk
            else:
                time.sleep(0.0002)
        return buf

    def flush_input(self):
        import termios
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def close(self):
        os.close(self.fd)


class FeetechBus:
    """The servo bus: ping / read / write / sync_read / sync_write of registers by name.

    Some USB adapters for half-duplex buses ECHO every byte sent. An echoed request is a valid packet too, and
    would be parsed as the servo's reply (a READ request's [addr, size] taken for a position). The first transfer
    detects the echo (the reply starts with the request) and strips it from then on."""

    def __init__(self, port, echo: bool | None = None):
        self.port, self.echo = port, echo

    def _txrx(self, pkt: bytes, expect_bytes: int) -> list:
        self.port.flush_input()
        self.port.write(pkt)
        if self.echo is None:                     # detect once: does the request come back first?
            data = self.port.read(len(pkt) + expect_bytes)
            self.echo = data.startswith(pkt)
            data = data[len(pkt):] if self.echo else data
        else:
            data = self.port.read((len(pkt) if self.echo else 0) + expect_bytes)
            data = data[len(pkt):] if self.echo and data.startswith(pkt) else data
        return parse_status(data) if expect_bytes else []

    def ping(self, servo_id: int) -> bool:
        return any(sid == servo_id for sid, _, _ in self._txrx(packet(servo_id, PING), 6))

    def read(self, servo_id: int, reg: str) -> int | None:
        addr, size = REG[reg]
        st = [s for s in self._txrx(packet(servo_id, READ, bytes([addr, size])), 6 + size) if s[0] == servo_id]
        if not st or len(st[0][2]) < size:
            return None
        return st[0][2][0] if size == 1 else from_u16(st[0][2])

    def write(self, servo_id: int, reg: str, value: int):
        addr, size = REG[reg]
        self._txrx(packet(servo_id, WRITE, bytes([addr]) + (bytes([value]) if size == 1 else u16(value))), 6)

    def sync_write(self, reg: str, values: dict):
        addr, size = REG[reg]
        params = bytes([addr, size]) + b"".join(bytes([i]) + (bytes([v]) if size == 1 else u16(v))
                                                for i, v in values.items())
        self._txrx(packet(0xFE, SYNC_WRITE, params), 0)       # broadcast: no answer

    def sync_read(self, reg: str, ids) -> dict:
        addr, size = REG[reg]
        ids = list(ids)
        st = self._txrx(packet(0xFE, SYNC_READ, bytes([addr, size]) + bytes(ids)), len(ids) * (6 + size))
        out = {}
        for sid, _, params in st:
            if sid in ids and len(params) >= size:
                out[sid] = params[0] if size == 1 else from_u16(params)
        return out


class MockBus:
    """Simulated servos for tests and dry runs: positions move toward their goals at a fixed speed."""

    def __init__(self, positions: dict, speed_ticks_per_s: float = 3000.0):
        self.pos = {i: float(p) for i, p in positions.items()}
        self.goal = dict(self.pos)
        self.torque = {i: 0 for i in positions}
        self.speed, self.t = speed_ticks_per_s, time.time()
        self.writes = []

    def _advance(self):
        now = time.time()
        step, self.t = self.speed * (now - self.t), now
        for i in self.pos:
            if self.torque[i]:
                d = self.goal[i] - self.pos[i]
                self.pos[i] += max(-step, min(step, d))

    def ping(self, servo_id):
        return servo_id in self.pos

    def read(self, servo_id, reg):
        self._advance()
        return int(round(self.pos[servo_id])) if reg == "present_position" else self.torque.get(servo_id, 0)

    def write(self, servo_id, reg, value):
        self._advance()
        self.writes.append((servo_id, reg, value))
        if reg == "torque_enable":
            self.torque[servo_id] = value
            if value:
                self.goal[servo_id] = self.pos[servo_id]
        elif reg == "goal_position":
            self.goal[servo_id] = float(value)

    def sync_write(self, reg, values):
        for i, v in values.items():
            self.write(i, reg, v)

    def sync_read(self, reg, ids):
        return {i: self.read(i, reg) for i in ids if i in self.pos}


# --- calibration --------------------------------------------------------------------------------------------------
@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    range_min: int
    range_max: int
    homing_offset: int = 0

    @property
    def mid(self) -> float:
        return (self.range_min + self.range_max) / 2.0


class Calibration:
    """ticks <-> joint radians for every motor of a robot (LeRobot calibration JSON or the same fields in YAML)."""

    def __init__(self, motors: dict, gripper: str | None = None, gripper_limits=(-0.17453, 1.74533)):
        self.motors = motors                       # name -> MotorCalibration
        self.gripper, self.gripper_limits = gripper, gripper_limits

    @classmethod
    def load(cls, path, gripper="gripper", gripper_limits=(-0.17453, 1.74533)):
        text = open(path, encoding="utf-8").read()
        if str(path).endswith((".yaml", ".yml")):
            import yaml
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        motors = {n: MotorCalibration(int(m["id"]), int(m.get("drive_mode", 0)), int(m["range_min"]),
                                      int(m["range_max"]), int(m.get("homing_offset", 0))) for n, m in data.items()}
        return cls(motors, gripper if gripper in motors else None, gripper_limits)

    def to_rad(self, name: str, ticks: float) -> float:
        m = self.motors[name]
        sign = -1.0 if m.drive_mode else 1.0
        if name == self.gripper:
            lo, hi = self.gripper_limits
            f = (ticks - m.range_min) / max(1, m.range_max - m.range_min)
            f = 1.0 - f if m.drive_mode else f
            return lo + f * (hi - lo)
        return sign * (ticks - m.mid) * RAD_PER_TICK

    def to_ticks(self, name: str, rad: float) -> int:
        m = self.motors[name]
        if name == self.gripper:
            lo, hi = self.gripper_limits
            f = (rad - lo) / (hi - lo)
            f = 1.0 - f if m.drive_mode else f
            t = m.range_min + f * (m.range_max - m.range_min)
        else:
            sign = -1.0 if m.drive_mode else 1.0
            t = m.mid + sign * rad / RAD_PER_TICK
        return int(round(min(max(t, m.range_min), m.range_max)))     # never outside the recorded range
