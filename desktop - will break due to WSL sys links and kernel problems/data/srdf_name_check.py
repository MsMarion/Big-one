"""Does srdf_for() give the SRDF the generated URDF's robot name? (MoveIt drops an SRDF with another name.)"""
import re
import sys

sys.path.insert(0, "/repo")
from pnp_core import robot_description as r  # noqa: E402

u = open("/tmp/pnp_generated/so101.urdf").read()
s = open("/repo/robots/so101/so101.srdf").read()
out = r.srdf_for(s, u)
print("matches:", [(m.start(), m.group(0)) for m in re.finditer(r'(<robot[^>]*?name=")[^"]*', s)])
print("result :", re.findall(r"<robot [^>]*>", out))
print("same object:", out == s)
print("ET name:", r.ET.fromstring(u).get("name"), "| re module:", r.re)
name = r.ET.fromstring(u).get("name")
print("direct:", re.findall(r"<robot [^>]*>", re.sub(r'(<robot[^>]*?name=")[^"]*', lambda m: m.group(1) + name, s, count=1)))
import inspect
print(inspect.getsourcefile(r.srdf_for), r.srdf_for.__code__.co_consts)
