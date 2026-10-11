"""The focus message (pnp_core/focus.py): object + optional "near here" hint, old plain names still accepted."""
from pnp_core.focus import focus_message, nearest, parse_focus


def test_plain_name_round_trip():
    assert focus_message("lego_2x4") == "lego_2x4"
    assert parse_focus("lego_2x4") == ("lego_2x4", None)
    assert parse_focus("  cracker_box \n") == ("cracker_box", None)


def test_hint_round_trip():
    name, xy = parse_focus(focus_message("lego_2x2", (0.2079, 0.0773)))
    assert name == "lego_2x2" and xy == (0.2079, 0.0773)


def test_bad_json_is_no_focus():
    assert parse_focus("{not json") == ("", None)
    assert parse_focus("") == ("", None)


def test_nearest_picks_the_hinted_instance():
    items = [(0.10, 0.00), (0.20, 0.05), (0.21, 0.06)]
    assert nearest(items, None) == items                       # no hint: all of them (the old behaviour)
    assert nearest(items, (0.205, 0.052)) == [(0.20, 0.05)]    # the nearest one only
    assert nearest(items, (0.50, 0.50)) == []                  # nothing within the radius
