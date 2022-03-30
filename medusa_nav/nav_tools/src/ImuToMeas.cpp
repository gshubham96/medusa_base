/* 
Developers: gshubham96 -> gshubham96@gmail.com
Description: Please check the documentation of this package for more info.
*/
#include "ImuToMeas.h"

/*
#######################################################################################################################
@.@ CONSTRUCTOR: put all dirty work of initializations here
#######################################################################################################################
*/
ImuToMeas::ImuToMeas(ros::NodeHandle *nodehandle, ros::NodeHandle *nodehandle_private):nh_(*nodehandle), nh_p_(*nodehandle_private) {
	ROS_INFO("in class constructor of ImuToMeas");
 	loadParams();
 	initializeSubscribers();
 	initializePublishers();
}

 /*
#######################################################################################################################
 @.@ Destructor
 #######################################################################################################################
 */
 ImuToMeas::~ImuToMeas() {

 	// +.+ shutdown publishers
	pub_meas_.shutdown();

 	// +.+ shutdown subscribers
 	sub_imu_.shutdown();

 	// +.+ shutdown node
 	nh_.shutdown();
 }

/*
#######################################################################################################################
@.@ Member Helper function to set up subscribers;
#######################################################################################################################
*/
void ImuToMeas::initializeSubscribers() {
	ROS_INFO("Initializing Subscribers for ImuToMeas");
	std::vector<std::string> ros_subs =
    	MedusaGimmicks::getParameters<std::vector<std::string>>(
        	nh_p_, "topics/subscribers");
	sub_imu_ = nh_.subscribe(ros_subs[0],2, &ImuToMeas::messageCallback,this);
}

/*
#######################################################################################################################
@.@ Member helper function to set up publishers;
#######################################################################################################################
*/
void ImuToMeas::initializePublishers() {
 	ROS_INFO("Initializing Publishers for ImuToMeas"); 	// ---> add publishers here
	std::vector<std::string> ros_pubs =
    	MedusaGimmicks::getParameters<std::vector<std::string>>(
        	nh_p_, "topics/publishers");
 	pub_meas_ = nh_.advertise<dsor_msgs::Measurement>(ros_pubs[0],2);
}

/*
#######################################################################################################################
@.@ Set frequency of the node default is 2
#######################################################################################################################
*/
double ImuToMeas::nodeFrequency()
{
 	double node_frequency;
 	nh_.param("node_frequency", node_frequency, 5.0);
 	ROS_INFO("Node will run at : %lf [hz]", node_frequency);
 	return node_frequency;
}

/*
#######################################################################################################################
@.@ Load the parameters
#######################################################################################################################
*/
void ImuToMeas::loadParams() {
 	ROS_INFO("Load the ImuToMeas parameters");
	frame_override = MedusaGimmicks::getParameters<std::string>(nh_p_, "frame_override", std::string());
}

/*
#######################################################################################################################
@.@ Callbacks Section / Methods
#######################################################################################################################
*/

/*
#######################################################################################################################
@.@ Callback Flag
#######################################################################################################################
*/
void ImuToMeas::messageCallback(const sensor_msgs::Imu& msg) {

	dsor_msgs::Measurement meas;
	// convert incoming sensor_msgs:LImu to dsor_msgs::Measurement format that filter understands
	meas = messageConvert(msg);

	if (frame_override != std::string()){
		meas.header.frame_id = frame_override;
	}

	pub_meas_.publish(meas);

}
/*
#######################################################################################################################
@.@ Logic Functions
#######################################################################################################################
*/
dsor_msgs::Measurement ImuToMeas::messageConvert(const sensor_msgs::Imu& msg) {

	// set sensor header
	dsor_msgs::Measurement meas ;
	meas.header.stamp = ros::Time::now();
	meas.header.frame_id = msg.header.frame_id;

	// convert quat to rpy (radians)
	tf2::Quaternion quat;
	tf2::fromMsg(msg.orientation, quat);
	tf2::Matrix3x3 m(quat);
	double r, p, y;
	m.getRPY(r, p, y);

	// init the sensor reading
	meas.value.push_back(r); meas.value.push_back(p); meas.value.push_back(y);
	meas.value.push_back(msg.angular_velocity.x);
	meas.value.push_back(msg.angular_velocity.y);
	meas.value.push_back(msg.angular_velocity.z);

	return meas;

}
/*
#######################################################################################################################
@.@ Main
#######################################################################################################################
*/
int main(int argc, char** argv)
{
 	// +.+ ROS set-ups:
 	ros::init(argc, argv, "tools_node"); //node name
 	// +.+ create a node handle; need to pass this to the class constructor
 	ros::NodeHandle nh;

 	ros::NodeHandle nh_p("~");

 	ROS_INFO("main: instantiating an object of type ImuToMeas");

 	// +.+ instantiate an ImuToMeas class object and pass in pointer to nodehandle for constructor to use
 	ImuToMeas tools(&nh,&nh_p);

 	// +.+ Added to work with timer -> going into spin; let the callbacks do all the work
 	ros::spin();

 	return 0;
 }

