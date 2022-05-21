#!/usr/bin/env python

from __future__ import division
import rospy
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, Bumper, HeadTouch
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

from CMAC import CMAC


class Central:

    def __init__(self, nn):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False
        self.key = ""
        self.BlobX = 0
        self.BlobY = 0

        pass

    def key_cb(self, data):
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        self.key = data.data

    def joints_cb(self, data):
        # rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    def bumper_cb(self, data):
        rospy.loginfo("bumper: " + str(data.bumper) + " state: " + str(data.state))
        if data.bumper == 0:
            self.stiffness = True
            print("bumper = 0: {:.2f}".format(self.stiffness))
        elif data.bumper == 1:
            self.stiffness = False
            print("bumper = 1: {:.2f}".format(self.stiffness))

    def touch_cb(self, data):
        rospy.loginfo("touch button: " + str(data.button) + " state: " + str(data.state))

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

                # print("The Centroid Coordinates of Largest Red Object: {}, {}".format(cx, cy)
                
                self.BlobX = cx
                self.BlobY = cy

                # Show centroid in image
                cv2.circle(cv_image, (cx,cy), 5, (255, 255, 0), -1)
            except ZeroDivisionError:
                pass
        
        cv2.imshow("image window",cv_image)
        cv2.imshow("mask window",mask)
        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly
            

    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
    def set_stiffness(self, value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name, Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)

    def set_joint_angles(self, HeadYaw_angle , Lelbow_yaw, Lelbow_roll, LShoulderPitch_angle, LShoulderRoll_angle):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append("HeadYaw") 
        joint_angles_to_set.joint_angles.append(HeadYaw_angle)

        joint_angles_to_set.joint_names.append("LElbowYaw")
        joint_angles_to_set.joint_angles.append(Lelbow_yaw)
        joint_angles_to_set.joint_names.append("LElbowRoll")
        joint_angles_to_set.joint_angles.append(Lelbow_roll)
        
        joint_angles_to_set.joint_names.append("LShoulderPitch")
        joint_angles_to_set.joint_angles.append(LShoulderPitch_angle)
        joint_angles_to_set.joint_names.append("LShoulderRoll")
        joint_angles_to_set.joint_angles.append(LShoulderRoll_angle)
        joint_angles_to_set.relative = False  # if true you can increment positions
        joint_angles_to_set.speed = 0.1  # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)

    def central_execute(self, weights, out):
        rospy.init_node('central_node', anonymous=True)  # initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states", JointState, self.joints_cb)
        rospy.Subscriber("bumper", Bumper, self.bumper_cb)
        rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)

        rate = rospy.Rate(3)  # sets the sleep time to 3ms
        
        self.set_stiffness(True) 
        while not rospy.is_shutdown():
                               
            # Get centroid of blob position to feed into network -- normalize !!
            X = np.array([np.around(self.BlobX / 320, 5), np.around(self.BlobY / 240, 5)])

            # Adjust shoulder position based on input from red blob
            L2, output = cb.calculate_output(X, weights, out)
            # un-normalize the angles
            LShoulderPitch = np.around(2 * 2.0857 * output[0] - 2.0857, 5) # LSHP -2.08 --- 2.0
            LShoulderRoll = np.around(-0.3142 + output[1] * (0.3142 + 1.3265), 5) # LSHR -0.314  1.326

            print("cx: {:.2f}, cy: {:.2f}".format(self.BlobX, self.BlobY))
            print("LShoulderPitch: {:.2f}, LShoulderRoll: {:.2f}".format(LShoulderPitch, LShoulderRoll))

            # Head and ELbow angles were stiffed to the given angles -- same angles as used for the training process

            self.set_joint_angles(0.47,-1.237, -1.079, LShoulderPitch, LShoulderRoll)

            rospy.sleep(1.5)            
            rate.sleep()

        self.set_stiffness(False)

def main():
    # Load training data
    dirname = os.path.dirname(os.path.abspath(__file__))

    data = np.loadtxt(os.path.join(dirname, 'train_150.csv'), delimiter=',')
    # read data
    LShoulderPitch = data[:, 0]
    LShoulderRoll = data[:, 1]
    cx = data[:, 2]
    cy = data[:, 3]

    # Reshape input and output data
    X_train = np.vstack((cx, cy))
    X_train = X_train.reshape(2, len(cx))

    # OUTPUT DATA IS THE LSHOULDER JOINT ANGLES
    y_train = np.vstack((LShoulderPitch, LShoulderRoll))
    y_train = y_train.reshape(2, len(LShoulderPitch))

    # CMAC structure
    n_inputs = X_train.shape[0]
    resolution = 50
    receptive = 5
    n_outputs = y_train.shape[0]

    # Hyperparameters
    epochs = 20
    lr = 0.5

    cb = CMAC(n_inputs, resolution, receptive, n_outputs, epochs, lr)

    # Loading the pre-trained weights
    weights = np.loadtxt(os.path.join(dirname, "weights/weights.dat"))
    out = np.loadtxt(os.path.join(dirname, "weights/out.dat"))
    weights = weights.reshape((n_outputs, resolution, resolution))

    return cb, weights, out


if __name__ == '__main__':
    cb, weights, out = main()
    central_instance = Central(cb)
    central_instance.central_execute(weights, out)
