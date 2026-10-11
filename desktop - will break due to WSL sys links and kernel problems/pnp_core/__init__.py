"""pnp_core: the robot-agnostic math of the pick & place pipeline.

Pure Python (numpy + OpenCV). No ROS, no Isaac Sim, so everything here can be unit-tested in
seconds and reused unchanged by the ROS nodes, the sim tools, and the real robot.

Modules
  geometry         rotations, rigid transforms, camera look-at frames (ROS optical vs USD camera)
  perception_math  depth -> 3D points, mask -> object point cloud, object center estimation
  grasp            top-down grasp / pre-grasp / place poses from an estimated object pose
  verification     did the grasp work? (finger width checks, placement error)
  catalog          typed access to config/objects.yaml, robot and camera configs
"""
