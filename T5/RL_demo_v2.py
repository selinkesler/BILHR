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
import os

from sklearn import tree

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from keras.datasets import mnist
import pickle

from matplotlib import pyplot
import matplotlib.pyplot as plt

BIN_LEG = 8
BIN_GK = 4

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
        self.postureProxy = ALProxy("ALRobotPosture", self.robotIP, self.PORT)
        self.train = False
        self.test = False
        self.BIN_LEG = 8
        self.BIN_GK = 4

        self.RHipRoll_actions = np.linspace(-0.5, -0.8, BIN_LEG) # Number hip roll actions

        self.val = 1

        self.action_dict = {"left": 0, "right": 1, "kick": 2}        

        # Define rewards
        self.goal_reward = 20 # Reward for scoring goal
        self.miss_reward_goalkeeper = -2 # Miss the goal as goal keeper catches it (so found the goal actually)
        self.miss_reward_outside = -20 # Miss the goal completly bad shooting outside ------- my own idea
        self.fall_reward = -20 
        self.action_penalty = -1 # Penalty for each action execution ---  really need that ? 

        # self.RHipRoll_actions = np.linspace(-0.5, -0.8, self.BIN_LEG) # Number hip roll actions
        self.RKneePitch_actions = np.array((1.5, -0.1)) # will only consist of 2 values -- init value and kick (will kick forward from the middle position)

        self.action_translations = [(-1, 0), (1, 0), (0, 1)] # action translations within quantized action space
        self.action_list = [0, 1, 2]
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

    def save_Q(self,data):
        # Write the array to disk
        with open('Q_table.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(data.shape))

            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')

                # Writing out a break to indicate different slices...
                outfile.write('# New slice\n')

    def load_Q(self):

        # Read the array from disk
        new_data = np.loadtxt('Q_table.txt')
        data = new_data.reshape((self.BIN_LEG,self.BIN_GK,len(self.action_list)))

        return data

    def get_current_state(self,joint_RHip_Roll,BlobX):

        print('ROOOOLLLL : ',joint_RHip_Roll)

        # Get leg state -- 10 possible states
        max_state = self.RHipRoll_actions[BIN_LEG-1]
        min_state = self.RHipRoll_actions[0]

        # joint_angles[15] = RHipRoll
        s1 = round((joint_RHip_Roll - min_state)/(max_state - min_state) * (BIN_LEG-1))
        # Dividing the range -0.5 to -0.8 in 10 bins :
         # -0.5 , -0.53, -0.56, -0.59, -0.62, -0.65, -0.68, -0.71, -0.74, -0.77, -0.8
         # Example : -0.6 is the 6. intervall  (beginning from right with 0 and counting +1 in each new intervall)


        # For goal keeper only the x position is relevant, as we only shoot on the floor and not tricky shots :D
        # BlobX alredy normalized before
        s2 = round(BlobX * (BIN_GK-1))

        return np.array((s1, s2))

    def get_reward(self, rew):

        if rew == 1:
            reward = self.goal_reward
        elif rew == 2:
            reward = self.miss_reward_goalkeeper
        elif rew == 3:
            reward = self.miss_reward_outside
        elif rew == 4 :
            reward = self.fall_reward
        elif rew == 5 :
            reward = self.action_penalty            


        return reward   

    def action_execution(self, action):
        self.a1 +=  self.action_translations[action][0] # a1 = LEFT-RIGHT
        self.a2 += self.action_translations[action][1] # a2 = KICK

        print('a1 : ', self.a1)
        print('a2 : ', self.a2)
        
        # Stop doing stupid things
        if self.a1 < 0:
            self.a1 = 0
        elif self.a1 > self.BIN_LEG - 1:
            self.a1 = self.BIN_LEG - 1
        
        # Get action for NAO execution
       # print(self.RHipRoll_actions)
        RHipRoll = self.RHipRoll_actions[self.a1]
        RKnee_pitch = self.RKneePitch_actions[self.a2]

        print('RROOL ACTION : ', RHipRoll)

        return RHipRoll, RKnee_pitch

    def test_action(self, s1, s2, Q):

        actions_allowed = []
        if s1 > 0:
            # if s1 is larger equal to 1, then it could still move right
            actions_allowed.append(self.action_dict["left"])

        if s1 < self.BIN_LEG - 1 : 
            # if s1 is smaller equal to 8, then it could still move left
            actions_allowed.append(self.action_dict["right"])


        actions_allowed.append(self.action_dict["kick"]) # always kick allowed, anytime you want :)
        actions_allowed = np.array(actions_allowed, dtype=int)

        print('actions_allowed : ', actions_allowed)

        Q_sa = Q[s1, s2, actions_allowed]

        # Action selection from Q with the max probability
        action = actions_allowed[np.argmax(Q_sa)]

        print("Action: {}".format(action))

        return action   

    def plot_reward(self, tot_reward, z):

        tot_reward = np.vstack(tot_reward)
        plt.plot(tot_reward)
        plt.xlabel('Training Epochs')
        plt.ylabel('Total Reward')
        plt.grid()
        plt.savefig("Cummulative_Reward-{}Eps.png".format(z), dpi=500)
        # plt.show()


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


    def set_joint_angles(self,LHip_roll, LAnkle_roll, LKnee_pitch,LAnkle_pitch, RKnee_pitch, RHip_roll, RHip_pitch, RAnkle_pitch, Head_pitch, Head_yaw ,LShoulder_roll, RShoulder_roll, LShoulder_pitch, RShoulder_pitch):
        print('joint_angles function')
        joint_angles_to_set = JointAnglesWithSpeed()

        # LEFT LEG joints
        joint_angles_to_set.joint_names.append("LHipRoll") # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(LHip_roll) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.joint_names.append("LAnkleRoll")
        joint_angles_to_set.joint_angles.append(LAnkle_roll)
        joint_angles_to_set.joint_names.append("LKneePitch")
        joint_angles_to_set.joint_angles.append(LKnee_pitch)
        joint_angles_to_set.joint_names.append("LAnklePitch")
        joint_angles_to_set.joint_angles.append(LAnkle_pitch)  


        # RIGHT LEG joints
        joint_angles_to_set.joint_names.append("RKneePitch")
        joint_angles_to_set.joint_angles.append(RKnee_pitch)
        joint_angles_to_set.joint_names.append("RHipRoll")
        joint_angles_to_set.joint_angles.append(RHip_roll)
        joint_angles_to_set.joint_names.append("RHipPitch")
        joint_angles_to_set.joint_angles.append(RHip_pitch)
        joint_angles_to_set.joint_names.append("RAnklePitch")
        joint_angles_to_set.joint_angles.append(RAnkle_pitch)

    
        joint_angles_to_set.joint_names.append("HeadPitch")
        joint_angles_to_set.joint_angles.append(Head_pitch)
        joint_angles_to_set.joint_names.append("HeadYaw")
        joint_angles_to_set.joint_angles.append(Head_yaw)

        
        joint_angles_to_set.joint_names.append("LShoulderRoll")
        joint_angles_to_set.joint_angles.append(LShoulder_roll)
        joint_angles_to_set.joint_names.append("RShoulderRoll")
        joint_angles_to_set.joint_angles.append(RShoulder_roll)
        joint_angles_to_set.joint_names.append("LShoulderPitch")
        joint_angles_to_set.joint_angles.append(LShoulder_pitch)
        joint_angles_to_set.joint_names.append("RShoulderPitch")
        joint_angles_to_set.joint_angles.append(RShoulder_pitch)
    

        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)


    def central_execute(self, env):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        rate = rospy.Rate(1) # sets the sleep time to 1ms       
        reward_list = []
        total_reward = 0

        z = 1

        self.set_stiffness(True)
        self.postureProxy.goToPosture("Stand", 1.0)

        self.val = input("Enter 1 for train, 2 for test : ")

        while int(self.val) == 1 : # train mode
            ################ set joint angles to start position ################
            # LHip_roll, LAnkle_roll, LKnee_pitch,LAnkle_pitch, RKnee_pitch, RHip_roll, RHip_pitch, RAnkle_pitch, Head_pitch, Head_yaw ,LShoulder_roll, RShoulder_roll, LShoulder_pitch, RShoulder_pitch
            self.set_joint_angles(-0.3, 0.4, -0.09, 0.16, 0.4, -0.6, -0.2, -0.2, 0.2, 0, -0.5, -1.3,  -0.4 ,1.5)
            rospy.sleep(2)
            self.set_joint_angles(-0.3, 0.4, -0.09, 0.16, 1.5, -0.6, -0.2, -0.2, 0.2, 0, -0.5, -1.3, -0.4 , 1.5)
            rospy.sleep(4)

            # get the starting state 
            joint_RHip_Roll = self.joint_angles[15]
            state = self.get_current_state(float(joint_RHip_Roll), float(self.BlobX))
            state = state.astype(int)

            self.a1 = state[0]
            self.a2 = 0

            self.KICK = False

            while not self.KICK:


                s1 = state[0]
                action = env.get_action(state)

                if action != 2:
                    RHipRoll, RKnee_pitch = self.action_execution(action)

                    # EXECUTE ACTION
                    self.set_joint_angles(-0.3, 0.4, -0.09, 0.16, RKnee_pitch, RHipRoll, -0.2, -0.2, 0.2, 0, -0.5, -1.3, -0.4 , 1.5)
                    
                    rospy.sleep(3)
                    joint_RHip_Roll_after = self.joint_angles[15]

                    state_next = self.get_current_state(joint_RHip_Roll_after, float(self.BlobX))                   
                    rew = 5
                    reward = self.get_reward(int(rew))

                else :     
                    
                    # Example showing a single target for one joint
                    names             = "RKneePitch"
                    targetAngles      = -0.1
                    maxSpeedFraction  = 0.8 # Using 20% of maximum joint speed
                    self.motionProxy.angleInterpolationWithSpeed(names, targetAngles, maxSpeedFraction)  

                    # Give the REWARD for the action
                    rew = input("Enter Reward : ")
                    # go out of While not KICK loop and will set the joint angles back to start position
                    self.KICK = True

                    reward = self.get_reward(int(rew))
                    total_reward += reward
                    reward_list.append(total_reward)

                    z += 1

                # Increment visits
                env.visits[state[0], state[1], action] += 1

                Q_upd = env.upd_model(state, action, reward, state_next)

                # Update state for the next (as the actual) state
                state = state_next.astype(int)

                if z % 2 == 0:
                    print("Sample : ", z)
                    print("total Reward : ", total_reward)
                    self.save_Q(Q_upd)
                    self.plot_reward(reward_list,z)

                if self.KICK == True : 
                    cont_st = input("Continue with new sample (1) or Stop train (2) or : ")

                    if int(cont_st) == 1:

                        # Reset the states
                        state = np.zeros(2)

                    elif int(cont_st) == 2:                       
                        sure_val = input("Sure to break Train ?? Press 1 to Continue Train and 2 to go the Test mode or 3 to BREAK ALL: ")

                        if int(sure_val) == 1:
                            self.val = 1

                        elif int(sure_val) == 2:
                            # SAVE the model!!! 
                            self.save_Q(Q_upd)
                            # self.plot_reward(reward_list)
                            print('going to test mode') 
                            self.KICK = True
                            self.val = 2

                        elif int(sure_val) == 3:
                            # SAVE the model!!! 
                            self.save_Q(Q_upd)
                            self.plot_reward(reward_list)
                            print('model saved and broke the process')

                            break
                

        while int(self.val) == 2: # test mode

            print('TEST MODE')            
            # need to load the saved enviroment with the weights
            Q = self.load_Q()

            # ONLY NEED TO GET THE ACTION
            # no need for updates and stuff.
            # LHip_roll, LAnkle_roll, LKnee_pitch,LAnkle_pitch, RKnee_pitch, RHip_roll, RHip_pitch, RAnkle_pitch, Head_pitch, Head_yaw ,LShoulder_roll, RShoulder_roll, LShoulder_pitch, RShoulder_pitch
            self.set_joint_angles(-0.3, 0.4, -0.09, 0.16, 0.4, -0.6, -0.2, -0.2, 0.2, 0, -0.5, -1.3,  -0.4 ,1.5)
            rospy.sleep(2)
            self.set_joint_angles(-0.3, 0.4, -0.09, 0.16, 1.5, -0.6, -0.2, -0.2, 0.2, 0, -0.5, -1.3, -0.4 , 1.5)
            rospy.sleep(4)

            # get the starting state 
            joint_RHip_Roll = self.joint_angles[15]
            state = self.get_current_state(float(joint_RHip_Roll), float(self.BlobX))
            state = state.astype(int)


            self.a1 = state[0]
            self.a2 = 0

            self.KICK = False

            while not self.KICK:

                s1 = state[0]
                s2 = state[1]
                action = self.test_action(s1, s2, Q)

                if action != 2:

                    RHipRoll, RKnee_pitch = self.action_execution(action)

                    # EXECUTE ACTION
                    self.set_joint_angles(-0.3, 0.4, -0.09, 0.16, RKnee_pitch, RHipRoll, -0.2, -0.2, 0.2, 0, -0.5, -1.3, -0.4 , 1.5) # last one to set -0.3 if feet get stucked
                    
                    rospy.sleep(3)

                    joint_RHip_Roll_after = self.joint_angles[15]

                    state_next = self.get_current_state(float(joint_RHip_Roll_after), float(self.BlobX))

                    # Update state for the next (as the actual) state
                    state = state_next.astype(int)

                else :     
                    
                    # Example showing a single target for one joint
                    names             = "RKneePitch"
                    targetAngles      = -0.1
                    maxSpeedFraction  = 0.8 # Using 20% of maximum joint speed
                    self.motionProxy.angleInterpolationWithSpeed(names, targetAngles, maxSpeedFraction)  

                    # go out of While not KICK loop and will set the joint angles back to start position
                    self.KICK = True

                    cont_st = input("Continue with new sample (1) or Stop (2)?")

                    if int(cont_st) == 1:

                        # Reset the states
                        state = np.zeros(2)

                    elif int(cont_st) == 2:

                        break
    

class Environment: 

    def __init__(self):

        # Define rewards
        self.goal_reward = 20 # Reward for scoring goal
        self.miss_reward_goalkeeper = -2 # Miss the goal as goal keeper catches it (so found the goal actually)
        self.miss_reward_outside = -20 # Miss the goal completly bad shooting outside ------- my own idea
        self.fall_reward = -20 
        self.action_penalty = -1 # Penalty for each action execution    

        self.BIN_LEG = 8
        self.BIN_GK = 4


        # Init actions
        self.action_dict = {"left": 0, "right": 1, "kick": 2}
        self.action_list = [0, 1, 2]

        self.action_trans = [(-1, 0), (1, 0), (0, 1)] 

        # Decision Trees Init
        self.T_State1 = tree.DecisionTreeClassifier()
        self.T_State2 = tree.DecisionTreeClassifier()
        self.T_Reward = tree.DecisionTreeClassifier()

        # Define quantized action space
        # Define action space -- in bins (divide the allowed action space intervall to the bins)
            # More bins, more precise but harder and longer to train
        self.RHipRoll_actions = np.linspace(-0.5, -0.8, self.BIN_LEG) # Number hip roll actions
        self.RKneePitch_actions = np.array((1.5, -0.1)) # will only consist of 2 values -- init value and kick (will kick forward from the middle position)

        # Init the states S1 and S2
            # S1 : LEG Position (RHipRoll and RHipPitch --- I guess pitch is the one to kick so it doenst really needs intervalls just min and max)
                # Roll Should be divided into bins
            # S2 : Position of the goalkeeper -- also should be divided into bins -- maybe relative to the size of the goal

        self.Sm1 = np.zeros(self.BIN_LEG)
        self.Sm2 = np.zeros(self.BIN_GK)
        

        # Init Visit count
        self.visits = np.zeros((self.BIN_LEG, 
                           self.BIN_GK,
                           len(self.action_list)))



        # Init inputs -- should work with RELATIVE information of the joints etc. 
        self.input = np.zeros(3) # will consist of s1,s2 and a
        self.d_S1 = np.array((0))
        self.d_S2 = np.array((0))
        self.d_R = np.array((0))

        # Init Q Function
        self.Q = np.zeros((self.BIN_LEG, 
                           self.BIN_GK,
                           len(self.action_list)))

        # Init Probability Transition P(s'/s,a) and R Function 
        self.Pm = np.zeros((self.BIN_LEG, 
                           self.BIN_GK,
                           len(self.action_list)))

        self.Rm = np.zeros((self.BIN_LEG, 
                           self.BIN_GK,
                           len(self.action_list)))


        # Init Learning param
        self.disc_fac = 0.001 # Discount factor

        self.action_translations = [(-1, 0), (1, 0), (0, 1)] # action translations within quantized action space

    def get_action(self, state):

        s1 = int(state[0]) 
        s2 = int(state[1])

        actions_allowed = []
        if s1 > 0:
            # if s1 is larger equal to 1, then it could still move right
            actions_allowed.append(self.action_dict["left"])

        if s1 < (self.BIN_LEG - 1): 
            # if s1 is smaller equal to 8, then it could still move left
            actions_allowed.append(self.action_dict["right"])

        actions_allowed.append(self.action_dict["kick"]) # always kick allowed, anytime you want :)

        actions_allowed = np.array(actions_allowed, dtype=int)

        print('actions_allowed : ', actions_allowed)

        Q_sa = self.Q[s1, s2, actions_allowed]

        # Action selection from Q with the max probability
        a_idx = np.argmax(Q_sa)
        action = actions_allowed[a_idx]

        print("Action: {}".format(action))

        return action    
    

    def add_experience(self, n, state, action, delta):
        # decision trees will be trained/fitted again with a larger dataset in each iteration
        if n == 1:

            x = np.append(state,np.array(action))
            self.input = np.vstack((self.input, x))
            self.d_S1 = np.append(self.d_S1, delta)

            self.T_State1 = self.T_State1.fit(self.input, self.d_S1)
        elif n == 2:

            self.d_S2 = np.append(self.d_S2, delta)
            self.T_State2 = self.T_State2.fit(self.input, self.d_S2)
        elif n == 3:
            print('input : ', self.input)
            self.d_R = np.append(self.d_R, delta)
            self.T_Reward = self.T_Reward.fit(self.input, self.d_R)
            print('d_RRRRRRRR : ', self.d_R)

    
    def get_trans_prop(self, s1, s2, a):
        # State change predictions
        deltaS1_pred = self.T_State1.predict([[a, s1, s2]])
        deltaS2_pred = self.T_State2.predict([[a, s1, s2]])
        state_change_pred = np.append(deltaS1_pred, deltaS2_pred)

        # Next state prediction
        state_pred = np.array((s1, s2)) + state_change_pred

        probab = self.T_State1.predict([[a, s1, s2]])        
        # print('probab : ', probab)

        deltaS1_prob = np.max(self.T_State1.predict_proba([[a, s1, s2]]))
        deltaS2_prob = np.max(self.T_State2.predict_proba([[a, s1, s2]]))
        P_deltaS = deltaS1_prob * deltaS2_prob

        return P_deltaS
    

    def upd_model(self, state, action, reward, state_):

        diff_1 = state[0] - state_[0]
        diff_2 = state[1] - state_[1]

        self.add_experience(1, state, action, diff_1)
        self.add_experience(2, state, action, diff_2)
        self.add_experience(3, state, action, reward)

        # in each iteration there will be new classes add to the labels of the tree
        # print('Classes : ' ,self.T_State1.classes_)

        for i in range(self.BIN_LEG):
            for j in range(self.BIN_GK):
                for k in range(3):
                    self.Pm[i, j, k] = self.get_trans_prop(i, j, k)
                    deltaR_pred = self.T_Reward.predict([[i, j, k]]) # REWARD PREDICTION
                    self.Rm[i, j, k] = deltaR_pred[0]

        # check_model
        exp = np.all(self.Rm[state[0], state[1], :] < 0) or np.all(self.Rm[state[0], state[1], :] > 0)

        # Updating the Q value 
        minvisits = np.min(self.visits)
        # print('minvisits : ', minvisits)

        for i in range(self.BIN_LEG):
            for j in range(self.BIN_GK):
                for k in range(3):

                    if exp and self.visits[i, j, k] == minvisits: 
                        self.Q[i, j, k] = self.goal_reward
                    else:
                        self.Q[i, j, k] = self.Rm[i, j, k]

                        for l in range(self.BIN_LEG):
                            for m in range(self.BIN_GK):

                                self.Q[i, j, k] += self.disc_fac * self.Pm[l, m, k] * np.max(self.Q[l, m, :])

        return self.Q


if __name__=='__main__':

    env = Environment()

    # Instantiate central class and start loop
    central_instance = Central()
    central_instance.central_execute(env)  

