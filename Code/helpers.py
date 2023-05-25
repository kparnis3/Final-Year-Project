import airsim
import numpy as np
import os
import tempfile
import pprint
import cv2
from gym import Env
import sys
from gym import spaces
import time
from PIL import Image
from matplotlib import pyplot as plt
import traceback
# generate random integer values
from random import seed
from random import randint
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

def camera_call():
    try:
        client = airsim.MultirotorClient()
        x = range(10)
        client.enableApiControl(True)
        client.armDisarm(True)
        print("Moving...") 
        client.moveToPositionAsync(-10, 5, -10, 5).join()
        seed(1)

        for n in x:
            # seed random number generator
            
            cord = [-10, randint(-2, 10), -10]
            print("Moving..."+str(cord)) 
            client.moveToPositionAsync(*cord, 5).join()

            print("Hovering...")
            client.hoverAsync().join()

            image_request = airsim.ImageRequest( 
            1, airsim.ImageType.DepthVis, True
            ) # camera_name, image_type, pixels_as_float=True

            responses = client.simGetImages([image_request])
            
            img1d = np.array(responses[0].image_data_float, dtype=np.float)
            img1d = np.array(np.clip(255 * 3 * img1d, 0, 255), dtype=np.uint8)
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
            image = Image.fromarray(img2d)
            image = np.array(image.resize((128, 72)).convert('L'))
            #cv2.imwrite('view.png', image)

            """""
            img1d = np.array(rawImage[0].image_data_float, dtype=np.float64)
            img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
            img2d = np.reshape(img1d, (rawImage[0].height, rawImage[0].width))
            image = Image.fromarray(img2d)

            im_final = np.array(image.resize((84, 84)).convert("L"))
            im_final = im_final.reshape([84, 84, 1])

            #img_rgb = img1d.reshape(image.height, image.width, 3)

            # original image is fliped vertically
            img_rgb = np.flipud(im_final)   
            """""

            image_path = "C:/Users/User/Desktop/Uni/Thesis/Code/TestImages"
            airsim.write_png(os.path.normpath(f'{image_path}/imageChanged' + str(n) + '.png'), image)
            print("System update: Saved depth image at path.")

            if (responses[0] == None):
                print("Camera is not returning image, please check airsim for error messages")
                sys.exit(0)
            else:
                time.sleep(0.5)
    
    except (KeyboardInterrupt):
        print("Keyboard input recieved.... terminating")
        pass

    except Exception: 
        traceback.print_exc()
        pass
    
    client.reset()
    client.armDisarm(False)

#camera_call()
