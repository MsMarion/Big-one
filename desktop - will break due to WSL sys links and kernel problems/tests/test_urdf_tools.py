from pnp_core.urdf_tools import clamp_joint_limits

URDF = """<robot name="r">
  <joint name="j1" type="revolute"><limit lower="-2.9007" upper="2.9007" effort="87" velocity="2.62"/>
    <safety_controller k_position="100" k_velocity="40" soft_lower_limit="-2.9007" soft_upper_limit="2.9007"/></joint>
  <joint name="j2" type="revolute"><limit lower="-1.0" upper="1.0" effort="87" velocity="2.62"/></joint>
</robot>"""


def test_clamp_narrows_never_widens():
    out, changes = clamp_joint_limits(URDF, {"j1": [-2.7437, 2.7437], "j2": [-3.0, 3.0]}, margin=0.02)
    assert 'lower="-2.7237"' in out and 'upper="2.7237"' in out
    assert 'soft_lower_limit="-2.7237"' in out
    assert 'lower="-1.0"' in out                     # j2: config is wider -> URDF unchanged
    assert len(changes) == 1 and changes[0].startswith("j1")
