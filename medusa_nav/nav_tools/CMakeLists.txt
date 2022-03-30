cmake_minimum_required(VERSION 3.1)
project(nav_tools)

find_package(catkin REQUIRED
 COMPONENTS
 std_msgs 
 auv_msgs
 dsor_msgs
 sensor_msgs 
 mavros_msgs
 medusa_msgs
 roscpp
 tf2
 tf2_ros
 medusa_gimmicks_library
)

find_package(GeographicLib REQUIRED)
include_directories(${GeographicLib_INCLUDE_DIR})

catkin_package(
 CATKIN_DEPENDS
 DEPENDS GeographicLib
)

# add_compile_options(-std=c++11) 
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_compile_options(-std=c++17 -Wall -O3) 

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

# usbl_to_pos
add_executable(usbl_to_pos src/UsblFix2Pos.cpp)
add_dependencies(usbl_to_pos ${catkin_EXPORTED_TARGETS})
target_link_libraries(usbl_to_pos ${catkin_LIBRARIES} ${GeographicLib_LIBRARIES})

# gnss_to_utm
add_executable(gnss_to_utm src/Gnss2Utm.cpp)
add_dependencies(gnss_to_utm ${catkin_EXPORTED_TARGETS})
target_link_libraries(gnss_to_utm ${catkin_LIBRARIES} ${GeographicLib_LIBRARIES})

# gnss_to_utm_outlier
add_executable(gnss_to_utm_outlier src/Gnss2UtmOutlier.cpp)
add_dependencies(gnss_to_utm_outlier ${catkin_EXPORTED_TARGETS})
target_link_libraries(gnss_to_utm_outlier ${catkin_LIBRARIES} ${GeographicLib_LIBRARIES})

# auvstate_to_mstate
add_executable(auvstate_to_mstate src/AuvState2mState.cpp include/LowPassFilter.h)
add_dependencies(auvstate_to_mstate ${catkin_EXPORTED_TARGETS})
target_link_libraries(auvstate_to_mstate ${catkin_LIBRARIES} ${GeographicLib_LIBRARIES})

# smooth_state
add_executable(smooth_state src/SmoothState.cpp)
add_dependencies(smooth_state ${catkin_EXPORTED_TARGETS})
target_link_libraries(smooth_state ${catkin_LIBRARIES} ${GeographicLib_LIBRARIES})

# imu_to_meas
add_executable(imu_to_meas src/ImuToMeas.cpp)
add_dependencies(imu_to_meas ${catkin_EXPORTED_TARGETS})
target_link_libraries(imu_to_meas ${catkin_LIBRARIES})

# pressure_to_depth
add_executable(pressure_to_depth src/FluidPressureToDepth.cpp)
add_dependencies(pressure_to_depth ${catkin_EXPORTED_TARGETS})
target_link_libraries(pressure_to_depth ${catkin_LIBRARIES})
