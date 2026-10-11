"""Runs with KiCad's own Python (pcbnew):  "<KiCad>/bin/python.exe" objsearch/kicad_flip.py BOARD.kicad_pcb FLIPS.json

Moves the listed footprints to the back side with KiCad's real Flip() (layers swapped, geometry mirrored, 3D model
flipped under the board), then applies each footprint's extra rotation. FLIPS.json: {"<ref>": <degrees>, ...}.
Standalone on purpose: KiCad's interpreter doesn't have objsearch's dependencies.
"""
import json
import sys

import pcbnew


def main(board_path, flips_path):
    with open(flips_path, encoding="utf-8") as f:
        flips = json.load(f)
    board = pcbnew.LoadBoard(board_path)
    done = []
    for fp in board.GetFootprints():
        ref = fp.GetReference()
        if ref not in flips:
            continue
        pos = fp.GetPosition()
        fp.Flip(pos, pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
        if flips[ref]:
            fp.Rotate(pos, pcbnew.EDA_ANGLE(float(flips[ref]), pcbnew.DEGREES_T))
        done.append(ref)
    pcbnew.SaveBoard(board_path, board)
    missing = sorted(set(flips) - set(done))
    print("OBJSEARCH_RESULT " + json.dumps({"flipped": len(done), "missing": missing}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
