import airsim
import numpy as np
import os
import gym
from gym import spaces
from PIL import Image, ImageStat
from matplotlib import pyplot as plt
import time
import random
from airsim.types import Pose,Quaternionr,Vector3r
import cv2
import threading
from threading import Thread
import glob
import os
import shutil
from decimal import Decimal

rewardConfig = {
    'collided': -100,
    'goal': 100,
    'timed': -100
}

TIME = 1

class drone_env_continuous(gym.Env):
    def __init__(self):
        super(drone_env_continuous, self).__init__()
        self.max_timestep= 250 
        self.goal_position = np.array([50.0, 0.0, -2.8], dtype=np.float64) #Unreal object in cm 53
        self.step_length = 1
        self.img_width = 150
        self.img_height = 150
        self.elapsed = 0
        self.start_time = 0
        self.timestep_count = 0
        self.goal_name = "Goal"
        self.goals = []
        self.sub_goal = 0
        self.VertPos = []
        self.HorzPos = []
        self.heading = Quaternionr(0, 0, 0, 0)


        # -- set the Observation Space --
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(self.img_height, self.img_width, 1), dtype=np.uint8),
            "velocity": spaces.Box(low=np.array([-np.inf for _ in range(3)]), 
                                   high=np.array([np.inf for _ in range(3)]),
                                   dtype=np.float64),
            "prev_relative_distance": spaces.Box(low=np.array([-np.inf for _ in range(3)]), 
                                            high=np.array([np.inf for _ in range(3)]),
                                            dtype=np.float64),
            "relative_distance": spaces.Box(low=np.array([-np.inf for _ in range(3)]), 
                                            high=np.array([np.inf for _ in range(3)]),
                                            dtype=np.float64)
            })

        # -- set internally state and info --

        self.state = {
            "image": np.zeros([self.img_height , self.img_width, 1], dtype=np.uint8),
            "velocity": np.zeros(3),
            "prev_relative_distance": np.zeros(3),
            "relative_distance": np.zeros(3)
        }

        self.info = {
            "prev_image": np.zeros([self.img_height , self.img_width, 1], dtype=np.uint8),
            "collision": False,
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "goal_position": self.goal_position,
            "goalreached": False
        }

        """
        Create the continuous space:

        1 - Move Foward or Back
        2 - Move Right or Left
        3 - Move Up or Down

        """
        self.action_space = spaces.Box(
            low=np.array([-1.0,-1.0,-1.0]),
            high=np.array([1.0,1.0,1.0]),
            dtype=np.float64
        )

        #  -- set drone client --
        self.drone = airsim.MultirotorClient() 
        self.drone.confirmConnection()

        #  -- set goal position --
        position = Vector3r(self.goal_position[0], self.goal_position[1], self.goal_position[2])
        pose = Pose(position, self.heading)
        self.drone.simSetObjectPose(self.goal_name, pose, True)

        self.setGoals()
        
        self.getParentObjPos()

        # -- set image request --
        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )   
   
    
    def getParentObjPos(self):
        VertNames = ["ParentVerticalFirstRow", "ParentVerticalSecondRow", "ParentVerticalThirdRow", "ParentVerticalFourthRow"]
        HorizNames = ["ParentHorizontalFirstRow", "ParentHorizontalSecondRow", "ParentHorizontalThirdRow", "ParentHorizontalFourthRow"]

        for x in range(len(VertNames)): #Assuming same size
            posV = self.drone.simGetObjectPose(VertNames[x])
            self.VertPos.append([posV, VertNames[x]])

            posH = self.drone.simGetObjectPose(HorizNames[x])
            self.HorzPos.append([posH, HorizNames[x]])
        self.orien = posH.orientation
        
    def setGoals(self):
        distance, _ = self.get_distance()
        sub_distance = distance / 4
        for _ in range(3):
            distance -= sub_distance
            self.goals.append(distance)
        self.goals.append(-99)

    def doAction(self, action):
        
        self.drone.startRecording()
        self.drone.moveByVelocityAsync(float(action[0]),
                                       float(action[1]),
                                       float(action[2]),
                                       TIME).join()
        self.drone.stopRecording()

        return
    
    
    def get_distance(self):

        dist = np.linalg.norm(self.info["position"]-self.goal_position)
        prev_dist = np.linalg.norm(self.info["prev_position"]-self.goal_position)
        #print(dist)
        #print(prev_dist)
       
        return dist, prev_dist
    
    def getRelativeDistance(self):

        r_x = np.linalg.norm(self.info["prev_position"][0]-self.goal_position[0])
        r_y = np.linalg.norm(self.info["prev_position"][1]-self.goal_position[1])
        r_z = np.linalg.norm(self.info["prev_position"][2]-self.goal_position[2])

        self.state["prev_relative_distance"] = np.array([r_x,r_y,r_z], dtype=np.float64)

        r_x = np.linalg.norm(self.info["position"][0]-self.goal_position[0])
        r_y = np.linalg.norm(self.info["position"][1]-self.goal_position[1])
        r_z = np.linalg.norm(self.info["position"][2]-self.goal_position[2])

        rel_dist = np.array([r_x,r_y,r_z], dtype=np.float64)
        return rel_dist
    
    def calculateReward(self, chosenAction): #figure out rewards
        done = False
        self.info["goalreached"] = False
        reward = 0

        distance, previous_distance = self.get_distance()
        reward = (previous_distance - distance) - np.linalg.norm(self.info["prev_position"]-self.info["position"])
        

        if distance <= self.goals[self.sub_goal]:
            print("Level: "+str(self.sub_goal))
            self.sub_goal += 1 
            reward = 20
               
        if(Decimal('-0.2') <= Decimal(str(chosenAction[0])) <= Decimal('0.2') 
            and 
            Decimal('-0.2') <= Decimal(str(chosenAction[1])) <= Decimal('0.2')
            and 
            Decimal('-0.2') <= Decimal(str(chosenAction[2])) <= Decimal('0.2')):
            reward = -1.5

        if self.info["collision"]:
            reward = rewardConfig['collided']
            done = True

        if self.state["relative_distance"][0] < 2:
            reward = rewardConfig['goal']
            self.info["goalreached"] = True
            print("System: Goal Reached.")
            done = True

        if self.timestep_count > self.max_timestep:
            print("System: Time Step Limit Reached")
            reward = rewardConfig['timed']
            done = True

        #print("Final reward: "+str(reward))
        #print(reward)
        
        return reward, done
    
    def step(self, chosenAction):
        self.timestep_count += 1
        self.drone.simPause(False)

        self.doAction(chosenAction)
        
        obs = self.getObservation()

        reward, done = self.calculateReward(chosenAction)

        #  -- Sometimes image bounces over obstacles once collision triggers --
        if done:
           mean1 = np.mean(self.state["image"])
           mean2 = np.mean(self.info["prev_image"])

           if mean2 > mean1:
            self.state["image"] = self.info["prev_image"]

        info = self.info

        return obs, reward, done, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.timestep_count = 0
        self.sub_goal = 0

        #  -- Randomise all our obstacles --
        self.randomiseObjects()

        #  -- Reset our action history --
        #self.state["action_history"] = -1 * np.ones(10, dtype=np.int8)
        self.drone.simPause(False)
        self.startFlight()
        self.drone.simPause(True)

        return self.getObservation()
    
    def randomiseObjects(self):
        for pos in self.VertPos:
            y = pos[0].position.y_val + np.random.uniform(-3,3)
            position = Vector3r(pos[0].position.x_val, y, pos[0].position.z_val)
            pose = Pose(position, self.heading)
            self.drone.simSetObjectPose(pos[1], pose, True)
        
        for pos in self.HorzPos:
            z = pos[0].position.z_val + np.random.uniform(-3,3)
            position = Vector3r(pos[0].position.x_val, pos[0].position.y_val, z)
            heading = Quaternionr(self.orien.x_val, self.orien.y_val , self.orien.z_val, self.orien.w_val) #90 degree rotation
            pose = Pose(position, heading)
            self.drone.simSetObjectPose(pos[1], pose, True)

    def startFlight(self):

        self.drone.reset()
        
        #  -- Randomise our drones starting location --
        ry = random.uniform(-9,9)
        rz = random.uniform(-7,0)

        position = airsim.Vector3r(0, ry, rz)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.drone.simSetVehiclePose(pose, True)
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True) 

        #print("System update: Drone has successfully been reset")
        self.drone.startRecording()
        self.drone.moveByVelocityAsync(0, 0, 0, 10).join()
        self.drone.stopRecording()

    def getImageObs(self):
        my_path = "C:/Users/User/Desktop/ThesisUnReal/TestImages2/"
        isExist = os.path.exists(my_path)

        if not isExist:
            os.makedirs(my_path)

        files = glob.glob(my_path + "*")

        imagelocation = files[0] + "/images/"
        mytime = 0
        start_time2 = time.time()
        while(True):
            time.sleep(0.001)
            if len(os.listdir(imagelocation)) >= 3:
                break
            if len(os.listdir(imagelocation)) < 3:
                mytime = time.time() - start_time2
                print("My Time:"+str(mytime))
                if(mytime > 1):
                    break

        imageL = glob.glob(imagelocation + "*")
        imageL.sort(key=os.path.getmtime)

        num = 0
        im_final = np.zeros([self.img_height, self.img_width, 1])

        for imL in imageL:
            im = None
            while(True):
                time.sleep(0.002)
                try:
                    im = cv2.imread(imL)
                except:
                    #time.sleep(0.005)
                    pass
                if im is not None:
                    break

            from PIL import Image

            im = np.expand_dims(im, axis=2)
            im = np.array(im, dtype=np.float64)
            img1d = 255 / np.maximum(np.ones(im.shape), im)
            img2d = np.reshape(img1d, (im.shape[0], im.shape[1]))
            
            image = Image.fromarray(img2d)
            image = image.rotate(180)
            image = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)

            #image_path = "TestImages"

            im_final = np.array(image.resize((150, 150)).convert("L"))
            
            im_final = im_final.reshape([150, 150, 1])
            #airsim.write_png(os.path.normpath(f'{image_path}/imageChanged{num}.png'), im_final)
            num+=1

        folder = my_path
        mytime = 0
        start_time2 = time.time()

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            flag = True
            while(True):
                try:
                    flag = True
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    #print('Failed to delete %s. Reason: %s' % (file_path, e))
                    time.sleep(0.001)
                    flag = False
                    mytime = time.time() - start_time2
                    if(mytime > 1):
                        break

                if(flag):
                    break

        return im_final

    def getObservation(self):

        self.drone.simPause(True)

        image = self.getImageObs()
        image_path = "TestImages"
        airsim.write_png(os.path.normpath(f'{image_path}/imageChanged.png'), image)
        self.info["prev_image"] = self.state["image"] 
        self.state["image"] = image

        self.drone_state = self.drone.getMultirotorState()

        kinematics = self.drone.getMultirotorState().kinematics_estimated


        self.info["collision"] = self.drone.simGetCollisionInfo().has_collided #check if drone has collided

        v_x = kinematics.linear_velocity.x_val
        v_y = kinematics.linear_velocity.y_val
        v_z = kinematics.linear_velocity.z_val

        self.state["velocity"] = np.array([v_x,v_y,v_z], dtype=np.float64)

        p_x = kinematics.position.x_val
        p_y = kinematics.position.y_val
        p_z = kinematics.position.z_val

        self.info["prev_position"] = self.info["position"] #get previous position
        self.info["position"] = np.array([p_x,p_y,p_z], dtype=np.float64) #get current position

        self.state["relative_distance"] = self.getRelativeDistance()

        return self.state

    def transformImage(self, responses):
        
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        
        from PIL import Image

        # Sometimes no image returns from api
        try:
            image = Image.fromarray(img2d)
            im_final = np.array(image.resize((self.img_width, self.img_height)).convert("L"))

            image_path = "TestImages"
            if not os.path.exists(image_path): #create directories if they dont already exist
                os.makedirs(image_path)

            airsim.write_png(os.path.normpath(f'{image_path}/imageChanged.png'), im_final)

            return im_final.reshape([self.img_height, self.img_width, 1])
        except:
            return np.zeros([self.img_height, self.img_width, 1])

    
    def disconnect(self):
        self.drone.enableApiControl(False)
        self.drone.armDisarm(False)
        print('System: Disconnected.')

        return