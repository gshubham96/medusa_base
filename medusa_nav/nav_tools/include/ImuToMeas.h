/* Developers: gshubham96 -> gshubham96@gmail.com */
#ifndef CATKIN_WS_IMUTOMEAS_H
#define CATKIN_WS_IMUTOMEAS_H

//ALWAYS need to include this 
#include <ros/ros.h> 

//some generically useful stuff to include...
#include <math.h>
#include <vector>
#include <stdlib.h>

// ROS messages
#include <sensor_msgs/Imu.h>
#include <dsor_msgs/Measurement.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// non-ros libraries
#include <medusa_gimmicks_library/MedusaGimmicks.h>



class ImuToMeas {
public:
	// #############################
 	// @.@ Constructor
 	// #############################
	ImuToMeas(ros::NodeHandle* nodehandle, ros::NodeHandle *nodehandle_private);

 	// #############################
 	// @.@ Destructor
 	// #############################
 	~ImuToMeas();

 	// #############################
 	// @.@ Public methods
 	// #############################
 	double nodeFrequency();

private:
 	ros::NodeHandle nh_, nh_p_;

 	// #####################
 	// @.@ Subsctibers
 	// #####################
 	ros::Subscriber sub_imu_;

 	// #####################
 	// @.@ Publishers
 	// #####################
 	ros::Publisher pub_meas_;

 	// ####################################################################################################################
 	// member variable: better than using globals; convenient way to pass data from a subscriber to other member functions
 	// member variables will retain their values even as callbacks come and go
 	// ####################################################################################################################

 	// +.+ Parameters from Yaml
	std::string frame_override;
 	// +.+ Handy variables

 	// +.+ Problem variables

 	// #######################################################################################
 	// @.@ Encapsulation the gory details of initializing subscribers, publishers and services
 	// #######################################################################################
 	void initializeSubscribers();
 	void initializePublishers();
 	void loadParams();

 	// #######################################################################################
 	// @.@ Callbacks declaration
 	// #######################################################################################
	void messageCallback(const sensor_msgs::Imu& msg);

public:
 	// #######################################################################################
 	// @.@ Auxilliary declaration
 	// #######################################################################################
	dsor_msgs::Measurement messageConvert(const sensor_msgs::Imu& msg);
};
#endif //CATKIN_WS_IMUTOMEAS_H
