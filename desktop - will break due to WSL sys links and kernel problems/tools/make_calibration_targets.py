"""Write the calibration targets as images: print-ready (300 DPI, exact size at 100 %) and used as sim textures.

    python3 /repo/tools/make_calibration_targets.py        -> models/calibration/{hand_board,table_board,apriltag_<id>}.png
    python3 /repo/tools/make_calibration_targets.py --print so101  -> data/print/so101_calibration_targets.pdf
        one US Letter page per target: the target at exact size + a 100 mm scale bar + what to measure
        (cells: so101 = the desk, so101_plate = the plate's tag set, fr3 = the FR3 table)
Also every scene's floor_grid texture (scene key floor_grid): models/calibration/<scene>_floor_grid.png, a 5 cm grid
with its origin at the robot base, cm labels and inch ticks (sim/scene.py lays it on the board).

Print at 100 % ("actual size", no fit-to-page), glue to something rigid and flat, then MEASURE one square / the tag
with calipers and put the measured size into config/calibration_targets.yaml (printers scale by a few tenths of %).
"""

import sys
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/repo")
from pnp_core.calibration import apriltag_image, board_image  # noqa: E402
from pnp_core.catalog import load_yaml  # noqa: E402

DPI = 300
OUT = Path("/repo/models/calibration")


def save(img, name):
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(OUT / name, dpi=(DPI, DPI))
    print(f"{OUT / name}: {img.shape[1]}x{img.shape[0]} px = {img.shape[1] / DPI * 25.4:.1f} x "
          f"{img.shape[0] / DPI * 25.4:.1f} mm at {DPI} DPI")


def main():
    if "--print" in sys.argv:
        cell = sys.argv[sys.argv.index("--print") + 1] if len(sys.argv) > sys.argv.index("--print") + 1 else "so101"
        return print_sheets(cell)
    for path in sorted(Path("/repo/config").glob("calibration_targets*.yaml")):          # every cell
        make(load_yaml(str(path)))
    for path in sorted(Path("/repo/config/scenes").glob("*.yaml")):                      # every floor grid
        scene = load_yaml(str(path))
        if scene.get("floor_grid"):
            floor_grid(scene)
    return 0


TARGETS = {"so101": "config/calibration_targets_so101.yaml", "so101_plate": "config/calibration_targets_so101_plate.yaml",
           "fr3": "config/calibration_targets.yaml"}


def floor_grid(scene, res=0.0005):
    """The board as a texture: a 5 cm grid (scene floor_grid.spacing_m) with its ORIGIN AT THE ROBOT BASE, labels in cm
    every label_every_m, inch ticks along the edges. Plate frame (sim/scene.py _flat_on_table, yaw 0): image columns
    run along +x (forward), rows along -y (row 0 = the board's left edge, y max)."""
    g, t = scene["floor_grid"], scene["table"]
    sx, sy = t["size"][0], t["size"][1]
    x0, y1 = t["center"][0] - sx / 2, t["center"][1] + sy / 2
    W, H = int(round(sx / res)), int(round(sy / res))
    img = Image.new("RGB", (W, H), (222, 200, 160))
    d = ImageDraw.Draw(img)
    col = lambda x: (x - x0) / res          # noqa: E731
    row = lambda y: (y1 - y) / res          # noqa: E731
    sp, lab = float(g.get("spacing_m", 0.05)), float(g.get("label_every_m", 0.10))
    f = _font(int(0.012 / res))
    for k in range(int(np.floor(x0 / sp)), int(np.ceil((x0 + sx) / sp)) + 1):
        x = k * sp
        major = abs(x / lab - round(x / lab)) < 1e-6
        d.line([(col(x), 0), (col(x), H)], fill=(60, 60, 60) if major else (140, 120, 95), width=3 if major else 1)
        if major:
            d.text((col(x) + 6, 6), f"x {x * 100:+.0f} cm", fill=(20, 20, 20), font=f)
            d.text((col(x) + 6, H - int(0.016 / res)), f"x {x * 100:+.0f} cm", fill=(20, 20, 20), font=f)
    for k in range(int(np.floor((y1 - sy) / sp)), int(np.ceil(y1 / sp)) + 1):
        y = k * sp
        major = abs(y / lab - round(y / lab)) < 1e-6
        d.line([(0, row(y)), (W, row(y))], fill=(60, 60, 60) if major else (140, 120, 95), width=3 if major else 1)
        if major:
            d.text((6, row(y) + 4), f"y {y * 100:+.0f} cm", fill=(20, 20, 20), font=f)
    # the base frame's axes through the origin: x forward (red), y left (green)
    d.line([(col(0), row(0)), (col(0.10), row(0))], fill=(200, 30, 30), width=6)
    d.line([(col(0), row(0)), (col(0), row(0.10))], fill=(30, 150, 30), width=6)
    d.text((col(0) + 10, row(0) + 10), "BASE (0, 0)", fill=(200, 30, 30), font=f)
    # inch ticks along all four edges (every inch; longer every 6", labelled every 12")
    inch = 0.0254
    for k in range(int(np.floor(x0 / inch)), int(np.ceil((x0 + sx) / inch)) + 1):
        x, n = k * inch, (8 if k % 12 == 0 else 5 if k % 6 == 0 else 3) * 0.002 / res
        for r0, r1 in ((0, n), (H - n, H)):
            d.line([(col(x), r0), (col(x), r1)], fill=(20, 20, 120), width=2)
        if k % 12 == 0:
            d.text((col(x) + 4, int(0.03 / res)), f'{k}"', fill=(20, 20, 120), font=f)
    for k in range(int(np.floor((y1 - sy) / inch)), int(np.ceil(y1 / inch)) + 1):
        y, n = k * inch, (8 if k % 12 == 0 else 5 if k % 6 == 0 else 3) * 0.002 / res
        for c0, c1 in ((0, n), (W - n, W)):
            d.line([(c0, row(y)), (c1, row(y))], fill=(20, 20, 120), width=2)
        if k % 12 == 0:
            d.text((int(0.085 / res), row(y) + 4), f'{k}"', fill=(20, 20, 120), font=f)   # clear of the cm label
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{scene['name']}_floor_grid.png"
    img.save(out)
    print(f"{out}: {W}x{H} px, {sx * 100:.1f} x {sy * 100:.1f} cm board, grid {sp * 100:.0f} cm, origin at the base")


def _font(size, bold=False):
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _page(img, title, lines, check, cut=None):
    """A US Letter page at DPI: the target at exact size, a 100 mm scale bar, text. cut: (top label, right label) ->
    a dashed cut line around the target's white border + where its edges point once it lies on the board."""
    W, H = int(8.5 * DPI), int(11 * DPI)
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    mm = DPI / 25.4
    d.text((int(15 * mm), int(12 * mm)), title, fill="black", font=_font(int(5.5 * mm), True))
    y = int(22 * mm)
    for ln in lines:
        d.text((int(15 * mm), y), ln, fill="black", font=_font(int(3.4 * mm)))
        y += int(5 * mm)
    tgt = Image.fromarray(img).convert("RGB")
    x0, y0 = (W - tgt.width) // 2, y + int(8 * mm)
    page.paste(tgt, (x0, y0))
    if cut:
        x1, y1 = x0 + tgt.width, y0 + tgt.height
        for k in range(0, tgt.width, int(4 * mm)):             # dashed: cut ON this line (keeps the white border)
            e = min(k + int(2 * mm), tgt.width)
            d.line([(x0 + k, y0 - 2), (x0 + e, y0 - 2)], fill="gray", width=2)
            d.line([(x0 + k, y1 + 1), (x0 + e, y1 + 1)], fill="gray", width=2)
            d.line([(x0 - 2, y0 + k), (x0 - 2, y0 + e)], fill="gray", width=2)
            d.line([(x1 + 1, y0 + k), (x1 + 1, y0 + e)], fill="gray", width=2)
        f = _font(int(3.4 * mm), True)
        d.text((x0, y0 - int(7 * mm)), "^ " + cut[0], fill="black", font=f)
        d.text((x1 + int(3 * mm), y0 + tgt.height // 2 - int(2 * mm)), "> " + cut[1], fill="black", font=f)
    # 100 mm scale bar with 10 mm ticks, below the target
    by = y0 + tgt.height + int(18 * mm)
    bx = (W - int(100 * mm)) // 2
    d.line([(bx, by), (bx + int(100 * mm), by)], fill="black", width=4)
    for k in range(11):
        h = int((4 if k % 5 else 7) * mm)
        x = bx + int(round(k * 10 * mm))
        d.line([(x, by - h), (x, by)], fill="black", width=3)
    d.text((bx, by + int(2 * mm)), "0", fill="black", font=_font(int(3.2 * mm)))
    d.text((bx + int(100 * mm) - int(8 * mm), by + int(2 * mm)), "100 mm", fill="black", font=_font(int(3.2 * mm)))
    yy = by + int(12 * mm)
    for ln in check:
        d.text((int(15 * mm), yy), ln, fill="black", font=_font(int(3.4 * mm)))
        yy += int(5 * mm)
    return page


def print_sheets(cell):
    """Print-ready PDF of a cell's calibration targets (docs/learning/22_jetson_d405.md: print, measure, enter)."""
    path = TARGETS.get(cell, "config/calibration_targets.yaml")
    t = load_yaml(path)
    px_per_m = DPI / 0.0254
    common = ["Print at 100 % / 'Actual size' (NOT 'Fit to page'), on matte paper, best printer quality.",
              "Glue it FLAT to something rigid (foam board, 3 mm plywood, acrylic). No wrinkles, no lamination glare."]
    pages = []
    use = {"table_board": "the WRIST D405's hand-eye calibration (lies flat on the desk)",
           "hand_board": "the fixed D435i's calibration (held in the SO-101's jaw), later"}
    for name in ("table_board", "hand_board"):
        if name not in t:
            continue
        b = t[name]
        img = board_image(b, px_per_m)
        pages.append(_page(img, f"{cell.upper()} {name.replace('_', ' ')}: ChArUco {b['squares_x']} x {b['squares_y']}",
                           [f"For {use[name]}.", *common,
                            f"Dictionary {b['dictionary']}, first marker id {b.get('first_id', 0)}."],
                           ["CHECK 1: the scale bar measures exactly 100 mm (else the printer scaled: reprint).",
                            "CHECK 2: measure across 4 squares with calipers / a steel ruler:",
                            f"   expected 4 x {b['square_m'] * 1000:.1f} = {4 * b['square_m'] * 1000:.1f} mm "
                            f"(markers {b['marker_m'] * 1000:.1f} mm).",
                            f"Enter the measured square size (m) as {name}.square_m and marker_m (scaled the same)",
                            f"in {path}."]))
    a = t["apriltags"]
    for tag in a["tags"]:
        img = apriltag_image(a["dictionary"], tag["id"], px=int(round(a["size_m"] * px_per_m)), white_border_frac=0.25)
        where = [f"Place at {tag.get('where', '')}: centre x {tag['sim_xy'][0] * 100:.1f} cm / "
                 f"{tag['sim_xy'][0] / 0.0254:.2f} in,",
                 f"y {tag['sim_xy'][1] * 100:.1f} cm / {tag['sim_xy'][1] / 0.0254:.2f} in from the robot base "
                 f"(x forward, y left)."]
        pages.append(_page(img, f"{cell.upper()} AprilTag {tag['id']} ({a['dictionary']})",
                           ["The drift monitor's fixed reference on the desk (outside the work band).", *where,
                            *common],
                           [f"CHECK: the scale bar measures 100 mm; the black square measures {a['size_m'] * 1000:.1f} mm.",
                            f"Enter the measured size as apriltags.size_m in {path}.",
                            "Cut on the dashed line (keep the white border: the detector needs it). Write the id on the back.",
                            "On the board: the TOP edge points to the robot's LEFT (+y), the RIGHT edge points away from",
                            "the robot (forward, +x). Position within ~1-2 cm is fine (it gets measured); flat + this turn matter."],
                           cut=(f"TOP (tag {tag['id']}): toward the robot's LEFT", "forward")))
    out = Path("/repo/data/print")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{cell}_calibration_targets.pdf"
    pages[0].save(f, save_all=True, append_images=pages[1:], resolution=DPI)
    for i, pg in enumerate(pages):                     # small previews (to look at, not to print)
        pg.resize((pg.width // 4, pg.height // 4)).save(out / f"{cell}_preview_{i + 1}.png")
    print(f"{f}: {len(pages)} pages (US Letter, {DPI} DPI)")
    return 0


def make(t):
    px_per_m = DPI / 0.0254
    for name in ("hand_board", "table_board"):
        if name in t:
            save(board_image(t[name], px_per_m), t[name].get("image", f"{name}.png"))
    a = t["apriltags"]
    for tag in a["tags"]:
        img = apriltag_image(a["dictionary"], tag["id"], px=int(round(a["size_m"] * px_per_m)), white_border_frac=0.25)
        save(img, f"{a.get('image_prefix', '')}apriltag_{tag['id']}.png")


if __name__ == "__main__":
    sys.exit(main())
