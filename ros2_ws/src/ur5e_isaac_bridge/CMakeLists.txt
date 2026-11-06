cmake_minimum_required(VERSION 3.22)
project(ur5e_isaac_bridge)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch rviz urdf scripts
  DESTINATION share/${PROJECT_NAME}
)

# ROS-side tools, runnable with `ros2 run ur5e_isaac_bridge <script>`.
# spawn_ur5e.py is not listed: it runs under Isaac Sim's Python.
install(
  PROGRAMS
    scripts/capture_wrist_images.py
    scripts/verify_projection.py
    scripts/check_camera_topics.py
    scripts/snap_tf_overlay.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
