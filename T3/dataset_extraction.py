#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys
from naoqi import ALProxy

class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False  
        self.key = 0
        self.bumper = False
        self.BlobX = 0
        self.BlobY = 0
        self.state = 0 
        self.robotIP = "10.152.246.171"
        self.PORT = 9559
        self.motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)

        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        self.LSPA = data.position[2] #left shoulder pitch angle
        self.LSRA = data.position[3] #left shoulder roll angle

        pass

    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
           # self.bumper = True
            self.stiffness = False
            self.bumper = True
            print('hehehhehehehh')

    def touch_cb(self,data):
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))

        self.key = data.button
        self.state = data.state

    
    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        # Extract red
        # we transform it into HSV
        image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        lower = np.array([161,155,84])
        upper = np.array([179,255,255])
        mask = cv2.inRange(image_hsv, lower, upper)

        # to binary image
        ret,thr = cv2.threshold(mask,127,255,0)

        # we find countours
        im2, contours, hierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas_list = []
        for c in contours:
            area = cv2.contourArea(c)
            areas_list.append(area)

        # Find the largest colour blob and calculate the position of its center in pixel coordinates 
        # Index of largest blob
        if len(areas_list) != 0:
            idx = np.argmax(areas_list)     

            # Calculate the moment of largest blob   
            M = cv2.moments(contours[idx])

            # Get x,y coordinate of centroid
            try:
                cx = int(M["m10"] / M["m00"])

                cy = int(M["m01"] / M["m00"])

                # print("The Centroid Coordinates of Largest Red Object: {}, {}".format(cx, cy))

                # CHECK THE IMAGE DIMENSIONS FOR NORMALIZATION!!!
                cx_norm = cx/320.0
                cy_norm = cy/240.0
                self.BlobX = cx_norm
                self.BlobY = cy_norm

                # Show centroid in image
                cv2.circle(cv_image, (cx,cy), 5, (255, 255, 0), -1)
            except ZeroDivisionError:
                pass
        
        cv2.imshow("image window",cv_image)
        cv2.imshow("mask window",mask)
        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly
            

    """
    # old image_to_cv funtion 

    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        cv2.imshow("image window",cv_image)
        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
    """

    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)


    def set_joint_angles(self,Lshoulder_pitch,Lshoulder_roll,Lelbow_yaw, Lelbow_roll, Rshoulder_pitch, Rshoulder_roll,Relbow_yaw, Relbow_roll):
        print('joint_angles function')
        joint_angles_to_set = JointAnglesWithSpeed()
        # LEFT ARM joints
        joint_angles_to_set.joint_names.append("LShoulderPitch") # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(Lshoulder_pitch) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.joint_names.append("LShoulderRoll")
        joint_angles_to_set.joint_angles.append(Lshoulder_roll)
        joint_angles_to_set.joint_names.append("LElbowYaw")
        joint_angles_to_set.joint_angles.append(Lelbow_yaw)
        joint_angles_to_set.joint_names.append("LElbowRoll")
        joint_angles_to_set.joint_angles.append(Lelbow_roll)
        

        # RIGHT ARM joints
        joint_angles_to_set.joint_names.append("RShoulderPitch")
        joint_angles_to_set.joint_angles.append(Rshoulder_pitch)
        joint_angles_to_set.joint_names.append("RShoulderRoll")
        joint_angles_to_set.joint_angles.append(Rshoulder_roll)
        joint_angles_to_set.joint_names.append("RElbowYaw")
        joint_angles_to_set.joint_angles.append(Relbow_yaw)
        joint_angles_to_set.joint_names.append("RElbowRoll")
        joint_angles_to_set.joint_angles.append(Relbow_roll)

        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)



    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        rate = rospy.Rate(1) # sets the sleep time to 1ms

        # Lshoulder_pitch,Lshoulder_roll,Lelbow_yaw, Lelbow_roll, Rshoulder_pitch, Rshoulder_roll,Relbow_yaw, Relbow_roll
        while not rospy.is_shutdown():
            # self.set_stiffness(self.stiffness)
            # self.set_stiffness(True)

            names  = ['LShoulderPitch','LShoulderRoll','HeadYaw', 'HeadPitch', 'LElbowYaw', 'LElbowRoll']

            # If only one parameter is received, this is applied to all joints''
            stiffnesses  = [0, 0, 0.9, 0.9, 0.9, 0.9]
            self.motionProxy.setStiffnesses(names, stiffnesses)         

            # need to set ELBOW and HEAD STIFF!!! 
            # HOW ?????

            # somehow in the stiffness function should be able to choose joints... 

            if self.key == 1 and self.state == 1 : # do we need data.state so that it wouln't save it

                #save the current joint angle settings for the shoulder
                LS_Pitch = self.LSPA 
                LS_Roll = self.LSRA               

                LS_Pitch_norm = np.around((LS_Pitch + 2.0857) / (2*2.0857), 5)
                LS_Roll_norm = np.around((LS_Roll + 0.3142) / (1.3265 + 0.3142), 5)

                ex = (np.asarray([[LS_Pitch_norm], [LS_Roll_norm], [self.BlobX], [self.BlobY]])).T

                #array = np.zeros((1,4))
                #array[0][0] = LS_Pitch_norm
                #array[0][1] = LS_Roll_norm
                #array[0][2] = self.BlobX
                #array[0][3] = self.BlobY

                # to save in csv file 
                with open('train_new.csv','a') as fd:
                    np.savetxt(fd, ex, delimiter=',',fmt='%1.3f')  


                print('SAVED!!') 

                # so that it wouldn't constantly save
                self.key = 0

                # self.set_joint_angles(0.7,0,0,-0.7, 0.7,0,0,0.7) -- nice home position angles (could be maybe useful)

                rospy.sleep(1.0)
            
            rate.sleep()

        self.set_stiffness(False)

    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
    rospy.spin()
