cmake_minimum_required(VERSION 3.16)
project(pnp_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DetectObjects.srv"
  "srv/RefinePose.srv"
  "action/PickPlace.action"
  DEPENDENCIES vision_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
