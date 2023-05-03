cmake_minimum_required(VERSION 3.0.2)
project(ros_osv)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    geometry_msgs
)


catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################



################################################
## Declare ROS dynamic reconfigure parameters ##
################################################



###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES trajectory_tools
 CATKIN_DEPENDS
 message_runtime
#  actionlib_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
${catkin_INCLUDE_DIRS}
)



#############
## Install ##
#############


catkin_install_python(
  PROGRAMS
  scripts/osv_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    )

## Mark other files for installation (e.g. launch and bag files, etc.)
# Install launch files
# install(DIRECTORY launch/
#     DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
#     )

#############
## Testing ##
#############

