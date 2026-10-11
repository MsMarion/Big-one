import math
import os

import pytest

from objsearch import kicad, track

BOARD = '''(kicad_pcb (version 20240108) (generator "pcbnew")
\t(footprint "1X03"
\t\t(layer "F.Cu")
\t\t(at 10 20 90)
\t\t(property "Reference" "J1" (at 0 0 0) (layer "UNDEFINED"))
\t\t(pad "1" thru_hole circle (at -2.54 0) (size 1.7 1.7) (drill 1) (layers "*.Cu"))
\t\t(pad "2" thru_hole circle (at 0 0) (size 1.7 1.7) (drill 1) (layers "*.Cu"))
\t\t(pad "3" thru_hole circle (at 2.54 0) (size 1.7 1.7) (drill 1) (layers "*.Cu"))
\t)
\t(footprint "WEIRD" (layer "F.Cu") (at 0 0) (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu")))
\t(zone (net 0) (net_name "") (layer "UNDEFINED") (uuid "z1") (polygon (pts (xy 0 0) (xy 1 0) (xy 1 1))))
\t(zone (net 0) (net_name "") (layer "F.Cu") (uuid "z2") (polygon (pts (xy 0 0) (xy 1 0) (xy 1 1))))
)
'''


def test_parse_and_sanitize():
    tree = kicad.parse(BOARD)[0]
    fps = kicad.children(tree, "footprint")
    assert [str(f[1]) for f in fps] == ["1X03", "WEIRD"]
    assert isinstance(fps[0][1], kicad.Str) and BOARD[fps[0].start] == "(" and BOARD[fps[0].end - 1] == ")"
    text, dropped = kicad.sanitize(BOARD)
    assert dropped == 1 and "UNDEFINED" not in text and '(uuid "z2")' in text and '(layer "Cmts.User")' in text


def _transform(pads, th, t):
    c, s = math.cos(th), math.sin(th)
    return {k: [(c * x - s * y + t[0], s * x + c * y + t[1]) for x, y in v] for k, v in pads.items()}


def test_fit_recovers_rotation_and_offset():
    lib = {"1": [(0, 0)], "2": [(0, 2.54)], "3": [(0, 5.08)], "4": [(7.62, 5.08)]}
    board = _transform(lib, math.radians(90), (-3.81, 1.27))
    th, t, rms, n = kicad.fit(lib, board)
    assert abs(math.degrees(th) - 90) < 1e-6 and abs(t[0] + 3.81) < 1e-6 and abs(t[1] - 1.27) < 1e-6
    assert rms < 1e-9 and n == 4
    assert kicad.fit({"1": [(0, 0)]}, {"1": [(1, 1)]}) is None  # needs 2+ pads


def test_fit_best_handles_aliases_and_pin_permutations():
    led_lib = {"1": [(-0.94, 0)], "2": [(0.94, 0)]}
    led_board = {"C": [(0, -1.05)], "A": [(0, 1.05)]}  # Eagle cathode/anode names
    th, _, rms, _ = kicad.fit_best(led_lib, led_board)
    assert rms < 0.2 and abs(abs(math.degrees(th)) - 90) < 1e-6
    # barrel-jack case: 9 Eagle pads (slots), 3 library pads, numbering differs -> permutation wins
    jack_lib = {"1": [(0.0, 0.0)], "2": [(-6.0, 0.0)], "3": [(-3.0, 4.7)]}
    jack_board = {"1": [(-5.08, 3.35)], "2": [(-0.08, 6.35)], "3": [(-0.08, 0.36)], "4": [(-5.08, 4.11)],
                  "5": [(-5.08, 2.59)], "6": [(-0.84, 0.36)], "7": [(0.69, 0.36)], "8": [(-0.84, 6.35)], "9": [(0.69, 6.35)]}
    th, _, rms, _ = kicad.fit_best(jack_lib, jack_board)
    assert rms < 0.2 and abs(math.degrees(th) - 90) < 0.5  # real Eagle pads: ~90.03 deg


def test_attach_models_places_and_reports(monkeypatch, tmp_path):
    step = tmp_path / "PinSocket_1x03.step"
    step.write_text("ISO-10303-21;")
    lib = {"pads": {"1": [(0, 0)], "2": [(0, 2.54)], "3": [(0, 5.08)]},
           "models": [("${KICAD10_3DMODEL_DIR}/X.3dshapes/PinSocket_1x03.wrl", [0, 0, 0], [0, 0, 0])], "name": "L:F"}
    monkeypatch.setattr(kicad, "load_footprint", lambda ref: lib)
    monkeypatch.setattr(kicad, "resolve_model", lambda p: str(step))
    text, rep = kicad.attach_models(kicad.sanitize(BOARD)[0], mapping={"1X03": ["L:F"]})
    assert rep["placed"] == 1 and rep["unmapped"] == {"WEIRD": ["?"]}  # refs, so flags can name the parts
    fp = kicad.children(kicad.parse(text)[0], "footprint")[0]
    m = kicad.child(fp, "model")
    off = kicad._nums(kicad.child(kicad.child(m, "offset"), "xyz"))
    rot = kicad._nums(kicad.child(kicad.child(m, "rotate"), "xyz"))
    # library pin column (along +y) must turn onto the board's row (along x): -90 deg, pad 1 at (-2.54, 0)
    th, t, _, _ = kicad.fit(lib["pads"], kicad._pads(fp))
    ys, rs = kicad._CONV
    assert abs(rot[2] - rs * math.degrees(th)) < 1e-3
    assert abs(off[0] - t[0]) < 1e-3 and abs(off[1] - ys * t[1]) < 1e-3
    assert str(m[1]).endswith("PinSocket_1x03.step")
    # a second pass doesn't add models twice
    assert kicad.attach_models(text, mapping={"1X03": ["L:F"]})[1]["placed"] == 0


@pytest.mark.skipif(not kicad.available(), reason="KiCad not installed")
def test_installed_library_lookup():
    hits = kicad.find("ESP32 WROOM 32")["models"]
    assert any("ESP32-WROOM-32" in h for h in hits)
    fp = kicad.load_footprint("Connector_BarrelJack:BarrelJack_Horizontal")
    assert set(fp["pads"]) == {"1", "2", "3"} and fp["models"]


@pytest.mark.bpy
def test_blender_normalize_fixes_units(tmp_path):
    """Real Blender: a 10x20x5 box authored in millimetre numbers -> metres, base on z=0, USD/GLB/OBJ out."""
    import trimesh
    from objsearch.blender import interface
    src = tmp_path / "box_mm.glb"
    trimesh.creation.box(extents=(10, 5, 20)).export(src)  # glTF y-up: height 5 along y
    res = interface.normalize(str(src), str(tmp_path / "out"), dims_mm=[20, 10, 5], name="box")
    assert res["scale_factor"] == 0.001 and res["dims_fit"] < 0.01
    assert sorted(res["dims_mm"]) == [5.0, 10.0, 20.0] and abs(res["dims_mm"][2] - 5.0) < 0.01
    for f in ("asset.usd", "model.glb", "mesh/mesh.obj", "thumb.png"):
        assert (tmp_path / "out" / f).exists()
    assert any("very low detail" in f for f in res["flags"])
