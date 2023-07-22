cmake_minimum_required(VERSION 3.0.2)
project(ros_plane_inspection)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
