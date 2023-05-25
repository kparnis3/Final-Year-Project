
import gym
import CustomEnv
import numpy as np

def test_env_random(): #used to test the enviroment with random actions
    env = gym.make("airsim-drone-continuous-v0")
    for episode in range(10):
        done = False
        score = 0
        img, obs = env.reset()

        while not done:
            action = env.action_space.sample()


            obs, reward, done, truncated, info = env.step(action)
            
            #print("Enter manual input")
            #input_a = input()
            #obs, reward, done, truncated, info = env.step(int(input_a))
            score += reward
        
        if episode % 10 == 0:
            print("Episode: {} Score: {}".format(str(episode), str(score)))
            
    env.reset()
    env.disconnect()
    return  


test_env_random()
#test_env_manual()


