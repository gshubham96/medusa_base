#!/usr/bin/env python

# -*- mode: Python -*-
""":"
# Bash code to check which python version to use
if [ "$ROS_DISTRO" == melodic ] ; then
    PYTHON_VERSION_TO_USE=python2
else
    PYTHON_VERSION_TO_USE=python3
fi

echo "$PYTHON_VERSION_TO_USE"

exec /usr/bin/$PYTHON_VERSION_TO_USE $0
":"""

# Basic ROS imports
import roslib 
roslib.load_manifest('dynamics_sim')
import rospy
import PyKDL
import sys
import rosnode

# import msgs
from nav_msgs.msg import Odometry           ## To publish vehicle pose
from dsor_msgs.msg import Thruster          ## To accept thruster input
from geometry_msgs.msg import WrenchStamped ## To accept collision forces

#import services
from std_srvs.srv import Empty

# More imports
from numpy import *
import tf

class Dynamics :
    
    def getConfig(self) :
        """ Load parameters from the rosparam server """
        rospy.init_node("dynamics_sim", anonymous=True)
        self.vehicle_name = rospy.get_param('~name')
        self.frame_id = self.vehicle_name
        
        # Get the world frame
        self.world_frame_id = rospy.get_param('world_frame')
        
        self.num_actuators = rospy.get_param("~dynamics/num_actuators")
        self.period = rospy.get_param('~dynamics/period')
        self.mass = rospy.get_param('~dynamics/mass')
        self.gravity_center = array(rospy.get_param('~dynamics/gravity_center'))
        self.g = rospy.get_param('~dynamics/g')
        self.radius = rospy.get_param('~dynamics/radius')
        self.ctf = rospy.get_param('~dynamics/ctf')
        self.ctb = rospy.get_param('~dynamics/ctb')
        self.actuators_tau = rospy.get_param('~dynamics/actuators_tau')
        self.actuators_maxsat= rospy.get_param('~dynamics/actuators_maxsat')
        self.actuators_minsat = rospy.get_param('~dynamics/actuators_minsat')
        self.actuators_gain = rospy.get_param('~dynamics/actuators_gain')
        self.dzv = rospy.get_param('~dynamics/dzv')
        self.dv = rospy.get_param('~dynamics/dv')
        self.dh = rospy.get_param('~dynamics/dh')
        self.density = rospy.get_param('~dynamics/density')
        self.damping = array(rospy.get_param('~dynamics/damping'))
        self.quadratic_damping = array(rospy.get_param('~dynamics/quadratic_damping'))
      
        self.am= rospy.get_param('~dynamics/allocation_matrix')
        self.am= array(self.am).reshape(self.num_actuators, 6)

        self.p_0 = array(rospy.get_param('~initial_pose'))
        self.v_0 = array(rospy.get_param("~initial_velocity"))
        
        # Currents data
        self.current_mean = array(rospy.get_param("~dynamics/current_mean"))
        self.current_sigma = array(rospy.get_param("~dynamics/current_sigma"))
        self.current_min = array(rospy.get_param("~dynamics/current_min"))
        self.current_max = array(rospy.get_param("~dynamics/current_max"))
        
        self.t_period = rospy.get_param("~dynamics/t_period")
        
    def s(self, x) :
        """ Given a 3D vector computes the 3x3 antisymetric matrix """
        # rospy.loginfo("s(): \n %s", x)
        ret = array([0.0, -x[2], x[1], x[2], 0.0, -x[0], -x[1], x[0], 0.0 ])
        return ret.reshape(3,3)

    def generalizedForce(self, du):
        """ Computes the generalized force as B*u, being B the allocation matrix and u the control input """
        # Allocation matrix is a [nx6] Matrix where n = no of thrusters
        # Each column of the matrix is constructed as follows: 
        # b(0:2, n) = Unit Allocation Vector representing contribution of thruster force in xyz direction
        # b(3:5, n) = Position of thruster w.r.t Centre of Mass of the vehicle 
        b=self.am
        
        # t = net generalized force
        t = zeros(6)
        for i in range(0,self.num_actuators):
            t1 = self.controlToForce(du[i], b[i][0:3], b[i][3:6])
            t = t + t1

        return t

    # u_n = control signal, r_n = position, a_n = allocation vector of thruster n 
    def controlToForce(self, u_n, a_n, r_n):
        ct = self.ctf
        if u_n < 0:
            ct = self.ctb
            
        # Calculate lump parameters for every thruster. Ref: https://journals.sagepub.com/doi/pdf/10.5772/56432, pg 3
        # f = net force by the thruster
        u_n = abs(u_n)
        f = (ct[0]*(u_n**2) + ct[1]*u_n + ct[2]*(sign(u_n)))
        # For kgf instead of N uncomment the next line. Pay attention to the ct variable
        # f = (ct[0]*(u_n**2) + ct[1]*u_n + ct[2]*(sign(u_n))) * self.g

        # The first three elements of the AM represents unit allocation vector (i^ + j^ + k^)
        # (Fx, Fy, Fz) = f dot (i^, j^, k^) i.e. find components of the force in 3d space
        F = zeros(3)
        F = dot(a_n, f)

        # The last three elements of the AM represents thruster position wrt COM
        # (Tx, Ty, Tz) = f cross (rx^, ry^, rz^), where T is torque, F is force
        T = zeros(3)
        T = cross(r_n, F)

        tau = zeros(6)
        tau[0:3] = F     # Assign Thruster Force
        tau[3:6] = T     # Assign Thruster Torque
        return tau

    def coriolisMatrix(self):
        s1 = self.s(dot(self.M[0:3,0:3], self.v[0:3]) + dot(self.M[0:3,3:6], self.v[3:6]))
        s2 = self.s(dot(self.M[3:6,0:3], self.v[0:3]) + dot(self.M[3:6,3:6], self.v[3:6])) 
        c = zeros((6, 6))
        c[0:3,3:6] = -s1
        c[3:6,0:3] = -s1
        c[3:6,3:6] = -s2
        return c
    
    def dumpingMatrix(self):
        # lineal hydrodynamic damping coeficients  
        Xu = self.damping[0]
        Yv = self.damping[1]
        Zw = self.damping[2]
        Kp = self.damping[3]
        Mq = self.damping[4]
        Nr = self.damping[5]
        
        # quadratic hydrodynamic damping coeficients
        Xuu = self.quadratic_damping[0]    #[Kg/m]
        Yvv = self.quadratic_damping[1]    #[Kg/m]
        Zww = self.quadratic_damping[2]    #[Kg/m]
        Kpp = self.quadratic_damping[3]    #[Kg*m*m]
        Mqq = self.quadratic_damping[4]    #[Kg*m*m]
        Nrr = self.quadratic_damping[5]    #[Kg*m*m]
    
        d = diag([Xu + Xuu*abs(self.v[0]), 
                  Yv + Yvv*abs(self.v[1]),
                  Zw + Zww*abs(self.v[2]),
                  Kp + Kpp*abs(self.v[3]),
                  Mq + Mqq*abs(self.v[4]),
                  Nr + Nrr*abs(self.v[5])])
        return d

    def gravity(self):
        """ Computes the gravity and buoyancy forces. Assumes a sphere model for now """
        #Weight and Flotability
        W = self.mass * self.g # [Kg]

        #If the vehicle moves out of the water the flotability decreases
        #FIXME: Assumes water surface at 0.0. Get this value from uwsim.
        if self.p[2] < 0.0: 
            r = self.radius + self.p[2]
            if r < 0.0:
                r = 0.0
        else :
            r = self.radius

        #FIXME: Write Floatability equation for medusa
        #TODO: either set as parameter, since different functions may be desired for different vehicles             
        #      or define common models and let the user choose one by the name
        #      Eventually let this part to bullet inside uwsim (HfFluid)

        F = ((4 * math.pi * pow(r,3))/3)*self.density*self.g 

        # if self.model_type == "sphere":         # Volume of Sphere
        #     V = (4 * math.pi * pow(r,3))/3
        # elif self.model_type == "cuboid":       # Volume of Cuboid         
        #     V = self.length * self.breadth * r
        # F = V * self.density * self.g 

        # gravity center position in the robot fixed frame (x',y',z') [m]
        zg = self.gravity_center[2]

        g = array([(W - F) * sin(self.p[4]),
                    -(W - F) * cos(self.p[4]) * sin(self.p[3]),
                    -(W - F) * cos(self.p[4]) * cos(self.p[3]),
                    zg*W*cos(self.p[4])*sin(self.p[3]),
                    zg*W*sin(self.p[4]),
                    0.0])

        return g
        
    def inverseDynamic(self) :
        """ Given the setpoint for each thruster, the previous velocity and the 
            previous position computes the v_dot """
        # du = self.thrustersDynamics(self.u)
        # t = self.generalizedForce(du)
        t = self.generalizedForce(self.u)
        c = self.coriolisMatrix()
        d = self.dumpingMatrix()
        g = self.gravity()
        c_v = dot((c-d), self.v)

        v_dot = dot(self.IM, (t-c_v-g+self.collisionForce)) # t-c_v-g+collisionForce
        v_dot = squeeze(asarray(v_dot)) #Transforms a matrix into an array
        self.collisionForce=[0,0,0,0,0,0]
        return v_dot
   
    def integral(self, x_dot, x, t) :
        """ Computes the integral o x dt """
        return (x_dot * t) + x
    
    def kinematics(self) :
        """ Given the current velocity and the previous position computes the p_dot """
        roll = self.p[3]
        pitch = self.p[4]
        yaw = self.p[5]
        
        rec = [cos(yaw)*cos(pitch), -sin(yaw)*cos(roll)+cos(yaw)*sin(pitch)*sin(roll), sin(yaw)*sin(roll)+cos(yaw)*cos(roll)*sin(pitch),
               sin(yaw)*cos(pitch), cos(yaw)*cos(roll)+sin(roll)*sin(pitch)*sin(yaw), -cos(yaw)*sin(roll)+sin(pitch)*sin(yaw)*cos(roll),
               -sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]
        rec = array(rec).reshape(3,3)
        
        to = [1.0, sin(roll)*tan(pitch), cos(roll)*tan(pitch),
              0.0, cos(roll), -sin(roll),
              0.0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)]
        to = array(to).reshape(3,3)
        
        p_dot = zeros(6)
        p_dot[0:3] = dot(rec, self.v[0:3])
        p_dot[3:6] = dot(to, self.v[3:6])
        return p_dot

    def updateThrusters(self, thrusters) :
        """Receives the control input, saturates each component to maxsat or minsat, and multiplies each component by the actuator gain"""
        #TODO: Check the size of thrusters.data
        t = array(thrusters.value)
        for i in range(size(t)):
            if abs(t[i]) > self.actuators_maxsat[i]:
                t[i] = sign(t[i]) * self.actuators_maxsat[i]
            elif abs(t[i]) < self.actuators_minsat[i] or isnan(t[i]):
                t[i] = 0
        self.u=t
        for i in range(self.num_actuators):
            self.u[i] = self.u[i]*self.actuators_gain[i]
        print(self.u)

    # simulates "realistic" thruster behavior by intriducing delay when large changes to the input signals are made     
    def thrustersDynamics(self, u):
        y = zeros(size(u))
        for i in range(size(u)):
            y[i] = (self.period * u[i] + self.actuators_tau[i] * self.y_1[i]) / (self.period + self.actuators_tau[i])
            
        self.y_1 = y
        return y
  
    def updateCollision(self, force) :
        self.collisionForce=[force.wrench.force.x,force.wrench.force.y,force.wrench.force.z,force.wrench.torque.x,force.wrench.torque.y,force.wrench.torque.z]        
    
    def pubPose(self, event):
        odom = Odometry()
        
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.frame_id

        odom.pose.pose.position.x = self.p[0]
        odom.pose.pose.position.y = self.p[1]
        odom.pose.pose.position.z = self.p[2]
        
        orientation = tf.transformations.quaternion_from_euler(self.p[3], self.p[4], self.p[5], 'sxyz')
        odom.pose.pose.orientation.x = orientation[0]
        odom.pose.pose.orientation.y = orientation[1]
        odom.pose.pose.orientation.z = orientation[2]
        odom.pose.pose.orientation.w = orientation[3]

        odom.twist.twist.linear.x = self.v[0]
        odom.twist.twist.linear.y = self.v[1]
        odom.twist.twist.linear.z = self.v[2]

        odom.twist.twist.angular.x = self.v[3]
        odom.twist.twist.angular.y = self.v[4]
        odom.twist.twist.angular.z = self.v[5]

        self.pub_pose.publish(odom)
     
        # Broadcast transform
        br = tf.TransformBroadcaster()
        br.sendTransform((self.p[0], self.p[1], self.p[2]), orientation, 
        rospy.Time.now(), self.world_frame_id, str(self.frame_id))
    
    def computeTf(self, tf):
        r = PyKDL.Rotation.RPY(math.radians(tf[3]), math.radians(tf[4]), math.radians(tf[5]))
        v = PyKDL.Vector(tf[0], tf[1], tf[2])
        frame = PyKDL.Frame(r, v)
        return frame

    def reset(self,req):
        self.v = self.v_0
        self.p = self.p_0
        return []
    
    def __init__(self):
        """ Simulates the dynamics of an AUV """

        # if len(sys.argv) != 6: 
        #   sys.exit("Usage: "+sys.argv[0]+" <namespace> <input_topic> <output_topic>")

        # Load dynamic parameters
        self.getConfig()
        rospy.init_node("dynamics_sim", anonymous=True)

        # Collision parameters
        self.collisionForce = [0,0,0,0,0,0]

        # self.altitude = -1.0 
        self.y_1 = zeros(self.num_actuators)
        
        # Init pose and velocity and period
        self.v = self.v_0
        self.p = self.p_0
        
        m = self.mass
        xg = self.gravity_center[0]
        yg = self.gravity_center[1]
        zg = self.gravity_center[2]
        
        # Inertia matrix of the rigid body
        Mrb = rospy.get_param("~dynamics/Mrb")
        Mrb = array(Mrb).reshape(6, 6)
             
        # Added Mass derivative
        Ma = rospy.get_param("~dynamics/Ma")
        Ma = array(Ma).reshape(6, 6) 
        
        self.M = Mrb + Ma    # mass matrix: Mrb + Ma
        self.IM = matrix(self.M).I
        # rospy.loginfo("Inverse Mass Matrix: \n%s", str(self.IM))
              
        # Init currents
        random.seed()
        self.e_vc = self.current_mean 
        self.e_vc = concatenate((self.e_vc, zeros((3,))))

	    # The number of zeros will depend on the number of actuators
        self.u = array(zeros(self.num_actuators)) # Initial thrusters setpoint

        self.thrusters_topic = rospy.get_param("~topics/subscribers/thrusters")
        self.external_force_topic = rospy.get_param("~topics/subscribers/external_force")
        self.position_topic = rospy.get_param("~topics/publishers/position")

        # Create publisher
        self.pub_pose= rospy.Publisher('~' + self.position_topic, Odometry, queue_size=1)
                
    	# Publish pose 
        rospy.Timer(rospy.Duration(self.t_period), self.pubPose)
        
        # Create Subscribers for thrusters and collisions
	    # TODO: set the topic names as parameters
        rospy.Subscriber(self.thrusters_topic, Thruster, self.updateThrusters)
        rospy.Subscriber('~' + self.external_force_topic, WrenchStamped, self.updateCollision)
        s = rospy.Service('/dynamics/reset',Empty, self.reset)
	
    def iterate(self):
        # Main loop operations
        self.v_dot = self.inverseDynamic()
        self.v = self.integral(self.v_dot, self.v, self.period)

        # compute currents
        random.seed()
        for i in range(3):
            self.e_vc[i] = random.normal(self.current_mean[i], self.current_sigma[i], 1)

        self.p_dot = self.kinematics() + self.e_vc
        self.p = self.integral(self.p_dot, self.p, self.period)


if __name__ == '__main__':
    try:
        dynamics = Dynamics() 
        rate_it = rospy.Rate(100)
        while not rospy.is_shutdown():
            dynamics.iterate()
            rate_it.sleep()
    except rospy.ROSInterruptException: pass
    
