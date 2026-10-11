from setuptools import setup

package_name = "pnp_perception"

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
    description="YOLO-seg + depth perception: 3D object positions for pick and place.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["detector_node = pnp_perception.detector_node:main",
                                      "obstacle_cloud_node = pnp_perception.obstacle_cloud_node:main",
                                      "camera_cloud_node = pnp_perception.camera_cloud_node:main",
                                      "scene_model_node = pnp_perception.scene_model_node:main",
                                      "pose_estimator_node = pnp_perception.pose_estimator_node:main",
                                      "sensor_model_node = pnp_perception.sensor_model_node:main",
                                      "camera_tf_node = pnp_perception.camera_tf_node:main",
                                      "calibration_monitor_node = pnp_perception.calibration_monitor_node:main",
                                      "wrist_refine_node = pnp_perception.wrist_refine_node:main",
                                      "hand_detector_node = pnp_perception.hand_detector_node:main"]},
)
