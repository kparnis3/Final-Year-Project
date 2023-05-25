import airsim
import random
import glob
import time
import cv2
import numpy as np
import time
import os
import shutil
#import warnings
#warnings.filterwarnings("ignore")


collision = False
drone = airsim.MultirotorClient() 
drone.confirmConnection()
drone.simPause(False)
ry = random.uniform(-9,9)
rz = random.uniform(-7,0)

position = airsim.Vector3r(0, ry, rz)
heading = airsim.utils.to_quaternion(0, 0, 0)
pose = airsim.Pose(position, heading)
drone.simSetVehiclePose(pose, True)
drone.enableApiControl(True)
drone.armDisarm(True) 
drone.moveByVelocityAsync(0, 0, 0, 3).join()

my_path = "C:/Users/User/Desktop/ThesisUnReal/TestImages2/"

for filename in os.listdir(my_path):
    file_path = os.path.join(my_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for _ in range(5_000):
# --------------------------------------------------------------------------------------
    my_path = "C:/Users/User/Desktop/ThesisUnReal/TestImages2/"
    
    isExist = os.path.exists(my_path)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(my_path)
    #print("The new directory is created!")

    drone.startRecording()
    #drone.simPause(False)
    drone.moveByVelocityAsync(1, 0, 0, 1).join()
    #drone.simPause(True)

    start_time = time.time()
    drone.stopRecording()

    # --------------------------------------------------------------------------------------
    """
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
            if(mytime > 1):
                break
            
    #print(len(os.listdir(imagelocation)))
    imageL = glob.glob(imagelocation + "*")
    imageL.sort(key=os.path.getmtime)
    
    #for file in imageL:
    #    print(file)

    num = 0
    im_final = np.zeros([150, 150, 1])
    for imL in imageL:
        im = None
        while(True):
            try:
                im = cv2.imread(imL)
            except:
                time.sleep(0.001)
                pass
            if im is not None:
                break

        from PIL import Image
        #print(im.shape)
        im = np.expand_dims(im, axis=2)
        #print(im.shape)
        #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #im = im.astype(np.float)
        im = np.array(im, dtype=np.float64)
        img1d = 255 / np.maximum(np.ones(im.shape), im)
        img2d = np.reshape(img1d, (im.shape[0], im.shape[1]))
        
        image = Image.fromarray(img2d)
        image = image.rotate(180)
        image = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)

        image_path = "TestImages"

        im_final = np.array(image.resize((150, 150)).convert("L"))

        #airsim.write_png(os.path.normpath(f'{image_path}/imageChanged{num}.png'), im_final)
        im_final = im_final.reshape([150, 150, 1])
        #print(im_final.dtype)
        airsim.write_png(os.path.normpath(f'{image_path}/imageChanged{num}.png'), im_final)
        #print(im_final.shape)
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
                mytime = time.time() - start_time2
                flag = False
                if(mytime > 1):
                    break

            if(flag):
                break
    

    #print ("My program took", time.time() - start_time, "to run")
    """
    # --------------------------------------------------------------------------------------
    
    start_time = time.time()
    image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        ) 

    responses = drone.simGetImages([image_request])

    img1d = np.array(responses[0].image_data_float, dtype=np.float64)
        
    img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        
    from PIL import Image

    # Sometimes no image returns from api

    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((150, 150)).convert("L"))

    image_path = "TestImages"
    if not os.path.exists(image_path): #create directories if they dont already exist
        os.makedirs(image_path)

    airsim.write_png(os.path.normpath(f'{image_path}/imageChanged.png'), im_final)

    im_final.reshape([150, 150, 1])
    print ("My program took", time.time() - start_time, "to run")
    
    
   # --------------------------------------------------------------------------------------
    collision = drone.simGetCollisionInfo().has_collided
    if collision:
        drone.simPause(False)
        collision = False
        drone.reset()
        
        #  -- Randomise our drones starting location --
        ry = random.uniform(-9,9)
        rz = random.uniform(-7,0)

        position = airsim.Vector3r(0, ry, rz)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        drone.simSetVehiclePose(pose, True)
        drone.enableApiControl(True)
        drone.armDisarm(True) 

        #print("System update: Drone has successfully been reset")
        drone.moveByVelocityAsync(0, 0, 0, 20).join()
        #time.sleep(3)
        
    
    