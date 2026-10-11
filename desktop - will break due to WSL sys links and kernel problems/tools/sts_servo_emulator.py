"""A byte-level emulator of a Feetech STS3215 servo bus on a pseudo-terminal (no hardware needed).

    python3 /repo/tools/sts_servo_emulator.py                  # prints the device path, e.g. /dev/pts/7
    python3 /repo/tools/sts_servo_emulator.py --echo           # + an adapter that echoes every request

Then point anything at that path instead of /dev/ttyACM0: tools/so101_servo_setup.py --port /dev/pts/7, or the
bringup (hardware:=real hw_port:=/dev/pts/7). Unlike the bridge's in-process mock bus, this exercises the REAL serial
code path: termios setup, packet bytes on a file descriptor, checksums, timeouts, half-duplex echo, SYNC_READ status
streams. It answers PING / READ / WRITE / SYNC_READ / SYNC_WRITE from a register table per servo, and moves each
servo toward its goal (torque on) at a servo-like speed, honoring Torque_Limit = 0 (= no torque).
Positions start at the middle of the example calibration's ranges (config/calibration/so101_desk/
so101_follower.example.json): the arm reads as near its zero pose.
"""

import argparse
import json
import os
import select
import sys
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnp_core.feetech import PING, READ, REG, SYNC_READ, SYNC_WRITE, WRITE, checksum, u16  # noqa: E402

ADDR = {a: (name, size) for name, (a, size) in REG.items()}


class Servo:
    def __init__(self, sid, pos):
        self.id = sid
        self.mem = {name: 0 for name in REG}
        self.mem.update(present_position=int(pos), goal_position=int(pos), torque_limit=1000,
                        present_voltage=120, present_temperature=31)
        self.pos = float(pos)

    def step(self, dt, speed=2500.0):
        if self.mem["torque_enable"] and self.mem["torque_limit"] > 0:
            d = self.mem["goal_position"] - self.pos
            self.pos += max(-speed * dt, min(speed * dt, d))
        self.mem["present_position"] = int(round(self.pos)) & 0xFFFF
        self.mem["moving"] = int(abs(self.mem["goal_position"] - self.pos) > 1 and self.mem["torque_enable"])

    def read(self, addr, size) -> bytes:
        out = b""
        a = addr
        while len(out) < size:
            name, sz = ADDR.get(a, (None, 1))
            v = self.mem.get(name, 0) if name else 0
            out += (bytes([v & 0xFF]) if sz == 1 else u16(v))
            a += sz
        return out[:size]

    def write(self, addr, data: bytes):
        a, i = addr, 0
        while i < len(data):
            name, sz = ADDR.get(a, (None, 1))
            if name:
                self.mem[name] = data[i] if sz == 1 else data[i] | (data[i + 1] << 8)
            a, i = a + sz, i + sz


def status(sid, params=b"", error=0):
    body = bytes([sid, len(params) + 2, error]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--echo", action="store_true")
    ap.add_argument("--calibration", default="config/calibration/so101_desk/so101_follower.example.json")
    ap.add_argument("--link", default="/tmp/so101_emulated_bus", help="also a stable symlink to the pty")
    a = ap.parse_args()
    cal = json.loads((Path(__file__).resolve().parents[1] / a.calibration).read_text())
    servos = {m["id"]: Servo(m["id"], (m["range_min"] + m["range_max"]) // 2) for m in cal.values()}
    master, slave = os.openpty()
    tty.setraw(master)
    path = os.ttyname(slave)
    try:
        if os.path.islink(a.link):
            os.remove(a.link)
        os.symlink(path, a.link)
    except OSError:
        pass
    print(f"emulated STS bus with servos {sorted(servos)} on {path} (link {a.link}){' with ECHO' if a.echo else ''}",
          flush=True)
    buf, last = b"", time.time()
    while True:
        r, _, _ = select.select([master], [], [], 0.002)
        now = time.time()
        for s in servos.values():
            s.step(now - last)
        last = now
        if not r:
            continue
        buf += os.read(master, 4096)
        while True:
            i = buf.find(b"\xff\xff")
            if i < 0 or len(buf) < i + 4:
                break
            length = buf[i + 3]
            end = i + 4 + length
            if len(buf) < end:
                break
            pkt, buf = buf[i:end], buf[end:]
            if checksum(pkt[2:-1]) != pkt[-1]:
                continue
            if a.echo:
                os.write(master, pkt)
            sid, instr, params = pkt[2], pkt[4], pkt[5:-1]
            reply = b""
            if instr == PING and sid in servos:
                reply = status(sid)
            elif instr == READ and sid in servos:
                reply = status(sid, servos[sid].read(params[0], params[1]))
            elif instr == WRITE and sid in servos:
                servos[sid].write(params[0], params[1:])
                reply = status(sid)
            elif instr == SYNC_WRITE:
                addr, size, rest = params[0], params[1], params[2:]
                for k in range(0, len(rest), size + 1):
                    if rest[k] in servos:
                        servos[rest[k]].write(addr, rest[k + 1:k + 1 + size])
            elif instr == SYNC_READ:
                addr, size = params[0], params[1]
                reply = b"".join(status(i, servos[i].read(addr, size)) for i in params[2:] if i in servos)
            if reply:
                os.write(master, reply)


if __name__ == "__main__":
    main()
