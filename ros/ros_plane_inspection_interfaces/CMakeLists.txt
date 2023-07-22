cmake_minimum_required(VERSION 3.0.2)
project(ros_plane_inspection_interfaces)

find_package(catkin REQUIRED COMPONENTS
  rospy
  #std_msgs
  visualization_msgs
  message_generation
)

# Generate services in the 'srv' folder
 add_service_files(
   FILES
   InspectPlane.srv
 )

# Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   visualization_msgs
 )
catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
