"""Reconstructed KiCad boards (objsearch.boardgen) and colour helpers."""
import os

import pytest

from objsearch import boardgen, kicad, parts

SPEC = os.path.join(os.path.dirname(__file__), "..", "..", "catalog", "boards", "u20cam-1080p-1.yaml")
STEP = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#10=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#11,#15),#23);
#15=MANIFOLD_SOLID_BREP('',#16);
#23=(GEOMETRIC_REPRESENTATION_CONTEXT(3));
ENDSEC;
END-ISO-10303-21;
"""


def test_tint_step_adds_colour_once(tmp_path):
    src, dst = tmp_path / "a.step", tmp_path / "b.step"
    src.write_text(STEP)
    assert parts.tint_step(str(src), str(dst), "#141414") == {"tinted": 1, "already_coloured": 0}
    out = dst.read_text()
    assert "COLOUR_RGB('',0.078431,0.078431,0.078431)" in out and "STYLED_ITEM('color'" in out
    assert out.index("MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION") < out.rindex("ENDSEC;")
    assert parts.tint_step(str(dst), str(tmp_path / "c.step"), "#ff0000") == {"tinted": 0, "already_coloured": 1}


def test_spec_to_board_offline(data_dir, tmp_path, monkeypatch):
    """The U20CAM spec builds a two-sided board KiCad accepts; downloads are faked, KiCad (library, flip) is real."""
    if not kicad.available():
        pytest.skip("KiCad not installed")
    holder = tmp_path / "h.kicad_mod"
    holder.write_text('(module X (layer F.Cu)\n (fp_line (start -6 -8) (end 6 -8) (layer F.SilkS) (width 0.5))\n'
                      ' (pad 1 thru_hole circle (at -10 0) (size 2.5 2.5) (drill 2) (layers *.Cu *.Mask))\n'
                      ' (pad 1 thru_hole circle (at 10 0) (size 2.5 2.5) (drill 2) (layers *.Cu *.Mask)))\n')
    step = tmp_path / "m.step"
    step.write_text(STEP)
    monkeypatch.setattr(boardgen, "_fetch", lambda url, sid: str(holder if url.endswith(".kicad_mod") else step))
    out = boardgen.build_pcb(boardgen.load(SPEC), str(tmp_path / "b.kicad_pcb"))
    text = open(out["path"], encoding="utf-8").read()
    tree = kicad.parse(text)[0]
    fps = kicad.children(tree, "footprint")
    side = {str(next(p[2] for p in kicad.children(f, "property") if p[1] == "Reference")): str(kicad.child(f, "layer")[1])
            for f in fps}
    spec = boardgen.load(SPEC)
    assert out["back_parts"] == len(spec["back_parts"]) and all(side[b["ref"]] == "B.Cu" for b in spec["back_parts"])
    assert side["J1"] == "F.Cu"  # the lens holder stays on the front
    j2 = next(f for f in fps if str(f[1]).startswith("Connector_JST:JST_PH_S5B"))
    assert [float(v) for v in kicad.child(j2, "at")[1:3]] == [-1.3, 12.8]  # pins = the VMPGG joints
    assert len(fps) == 4 + 1 + 7 + 1 + len(spec["back_parts"])  # holes, holder, reserved pads, VMPGG labels, back  # 4 holes, holder, connector, 7 reserved, VMPGG
    assert '(copper_finish "ENIG")' in text and "#151515" in text and "REF**" not in text
    assert "${KICAD" not in text  # library model paths resolved
    # attach_models must leave it alone: holes non-physical, reserved pads DNP, real parts already have models
    _, rep = kicad.attach_models(text)
    assert rep["placed"] == 0 and not rep["unmapped"] and not rep["approximate"]
