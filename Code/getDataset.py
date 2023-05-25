import gym
import CustomEnv
import airsim
import os
from collections import deque

def test_env(): 
    env = gym.make("airsim-drone-dynamic-v0")
    image_path_col = "Dataset3/Collision"
    image_path_nocol = "Dataset3/NonCollision"
    counter_col = 0
    counter_nocol = 0
    for ep in range(500):
        print("Episode:" +str(ep))
        done = False
        stack = []
        obs, info  = env.reset()
        
        #print(info["goalreached"])
        stack.append(obs["image"])
        
        while not done:
            obs, _, done, _, _ = env.step(0)
                     
            stack.append(obs["image"])
            if done:
                reached = info["goalreached"]
                if reached:
                    while len(stack) != 0: #rest are threated as collision images
                        image = stack.pop()
                        airsim.write_png(os.path.normpath(f'{image_path_nocol}/nocol{counter_nocol:04}.png'), image)
                        counter_nocol += 1
                else:
                    for _ in range(3): #pop three most recent images, our collision images
                        if len(stack) == 0: #stack is empty
                            break
                        imageC = stack.pop()
                        airsim.write_png(os.path.normpath(f'{image_path_col}/col{counter_col:04}.png'), imageC)
                        counter_col += 1
                    while len(stack) != 0: #rest are threated as collision images
                        image = stack.pop()
                        airsim.write_png(os.path.normpath(f'{image_path_nocol}/nocol{counter_nocol:04}.png'), image)
                        counter_nocol += 1
            
    env.reset()
    return  

test_env()

    