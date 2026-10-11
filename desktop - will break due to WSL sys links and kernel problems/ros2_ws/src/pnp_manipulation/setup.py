from setuptools import setup

package_name = "pnp_manipulation"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnp",
    maintainer_email="noreply@example.com",
    description="Pick and place action server (state machine over MoveIt 2, ros2_control and perception).",
    license="Apache-2.0",
    entry_points={"console_scripts": ["pick_place_node = pnp_manipulation.pick_place_node:main",
                                      "so101_bridge = pnp_manipulation.so101_bridge:main",
                                      "servo_error_node = pnp_manipulation.servo_error_node:main",
                                      "workcell_node = pnp_manipulation.workcell_node:main",
                                      "wrist_map_node = pnp_manipulation.wrist_map_node:main"]},
)
