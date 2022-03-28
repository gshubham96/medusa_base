cmake_minimum_required(VERSION 2.8.3)
project(dynamics_sim)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  medusa_msgs
  nav_msgs
  roscpp
  std_msgs
  message_generation
)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

catkin_python_setup()

add_message_files(
	DIRECTORY msg
	FILES
	ThrustersData.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/dynamics.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)





