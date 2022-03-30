/* 
Developers: gshubham96 -> gshubham96@gmail.com
Description: Please check the documentation of this package for more info.
*/
#include "FluidPressureToDepth.h"

/*
#######################################################################################################################
@.@ CONSTRUCTOR: put all dirty work of initializations here
#######################################################################################################################
*/
FluidPressureToDepth::FluidPressureToDepth(ros::NodeHandle *nodehandle, ros::NodeHandle *nodehandle_private):nh_(*nodehandle), nh_p_(*nodehandle_private) {
	ROS_INFO("in class constructor of FluidPressureToDepth");
 	loadParams();
 	initializeSubscribers();
 	initializePublishers();

	init_mode = true;
	count = 0;
}

 /*
#######################################################################################################################
@.@ Destructor
#######################################################################################################################
*/
FluidPressureToDepth::~FluidPressureToDepth() {

 	// +.+ shutdown publishers
	pub_meas_.shutdown();

 	// +.+ shutdown subscribers
 	sub_pres_.shutdown();

 	// +.+ shutdown node
 	nh_.shutdown();
}

/*
#######################################################################################################################
@.@ Member Helper function to set up subscribers;
#######################################################################################################################
*/
void FluidPressureToDepth::initializeSubscribers() {
	ROS_INFO("Initializing Subscribers for FluidPressureToDepth");
	std::vector<std::string> ros_subs =
    	MedusaGimmicks::getParameters<std::vector<std::string>>(
        	nh_p_, "topics/subscribers");
	sub_pres_ = nh_.subscribe(ros_subs[0],2, &FluidPressureToDepth::messageCallback,this);
}

/*
#######################################################################################################################
@.@ Member helper function to set up publishers;
#######################################################################################################################
*/
void FluidPressureToDepth::initializePublishers() {
 	ROS_INFO("Initializing Publishers for FluidPressureToDepth"); 	// ---> add publishers here
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
double FluidPressureToDepth::nodeFrequency()
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
void FluidPressureToDepth::loadParams() {
 	ROS_INFO("Load the FluidPressureToDepth parameters");
	threshold = MedusaGimmicks::getParameters<int>(nh_p_, "threshold", 10);
	density = MedusaGimmicks::getParameters<double>(nh_p_, "density", 997.0474);
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
void FluidPressureToDepth::messageCallback(const sensor_msgs::FluidPressure& msg) {

	// Initialize zero depth at startup
	if (init_mode == true){
		pressure0 += msg.fluid_pressure;
		count++;

		if (count == threshold){
			pressure0 = pressure0/threshold;
			init_mode = false;
		}
	}

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
dsor_msgs::Measurement FluidPressureToDepth::messageConvert(const sensor_msgs::FluidPressure& msg) {

	// set sensor header
	dsor_msgs::Measurement meas ;
	meas.header.stamp = ros::Time::now();
	meas.header.frame_id = msg.header.frame_id;

	// convert pressure to depth
	// depth = [pressure(depth) - pressure(surface)] / [rho * g]
	double depth;
	depth = (msg.fluid_pressure - pressure0) / (density*G);

	// init the sensor reading
	meas.value.push_back(depth);
	meas.noise.push_back(msg.variance); 

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

 	ROS_INFO("main: instantiating an object of type FluidPressureToDepth");

 	// +.+ instantiate an FluidPressureToDepth class object and pass in pointer to nodehandle for constructor to use
 	FluidPressureToDepth tools(&nh,&nh_p);

 	// +.+ Added to work with timer -> going into spin; let the callbacks do all the work
 	ros::spin();

 	return 0;
 }

