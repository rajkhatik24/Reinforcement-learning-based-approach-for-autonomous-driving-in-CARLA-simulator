#!/usr/bin/env python
import carla
import glob
import os
import sys
import random
import time
import numpy as np
import math
import cv2
from collections import deque
import torch.nn as nn
import gym
from gymnasium import spaces
import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random



SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 30
MIN_REWARD = -200

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_Camera = None


    def __init__(self):
    	
        self.action_space = spaces.Discrete(6)
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world('Town05')
        print(f"Connected to {self.world.get_map().name}")
        print("---------------")
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.num_frames = 8
        self.frame_buffer = deque(maxlen=self.num_frames)
        print(self.model_3)



    def processing(self,image):
	    resized = cv2.resize(image,(84, 84),interpolation=cv2.INTER_AREA)
	    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
	    for i in range(8):
	        self.frame_buffer.append(gray)
	    if len(self.frame_buffer) == self.num_frames:
	        stacked_frames = np.stack(self.frame_buffer, axis=0)
	        stacked_frames = stacked_frames / 255.0 
	        stacked_frames = stacked_frames[np.newaxis, :]
	    return stacked_frames


    def reset(self):

        self.collision_hist = []
        self.actor_list =[]
        #self.spawn_points = map.get_spawn_points()
        #self.random_spawn_point = random.choice(spawn_points)
        #self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.spawn_point = carla.Transform(carla.Location(x=-220.0, y=0.0, z=2.0), carla.Rotation(yaw=180))
        #self.actor_transform = carla.Transform(self.random_spawn_point.location, self.random_spawn_point.rotation)
        self.vehicle = self.world.spawn_actor(self.model_3,self.spawn_point)
        self.actor_list.append(self.vehicle)


        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x",f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y",f"{self.im_height}")        
        self.rgb_cam.set_attribute("fov",f"110")


        transform = carla.Transform(carla.Location(x = 2.5,z = 0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam,transform,attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))


        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake =0.0))
        time.sleep(0.5)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform,attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_Camera is None:
            time.sleep(0.01) 
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle =0.0, brake = 0.0))
        return self.front_Camera

    def collision_data(self,event):
        self.collision_hist.append(event)

    def process_img(self,image):
    	i = np.array(image.raw_data)
    	i2 = i.reshape((self.im_height,self.im_width,4))
    	i3 = i2[:,:,:3]
    	if self.SHOW_CAM:
    		cv2.imshow("",i3)
    		cv2.waitKey(1)
    	self.front_Camera = i3

    def step(self,action):
        collided = False
        self.reward = 0
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle= 1.0, steer = -1*self.STEER_AMT))
        elif action==1:
            self.vehicle.apply_control(carla.VehicleControl(throttle= 1.0, steer = 0*self.STEER_AMT))
        elif action ==2:
            self.vehicle.apply_control(carla.VehicleControl(throttle= 1.0, steer = 1*self.STEER_AMT))
        elif action ==3:
            self.vehicle.apply_control(carla.VehicleControl(throttle= 0.5, steer = -1*self.STEER_AMT))
        elif action ==4:
            self.vehicle.apply_control(carla.VehicleControl(throttle= 0.5, steer = 0*self.STEER_AMT))
        elif action ==5:
            self.vehicle.apply_control(carla.VehicleControl(throttle= 0.5, steer = 1*self.STEER_AMT))


        v = self.vehicle.get_velocity()
        kmh = int(3.6 *math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            collided = True
            done = True
            self.reward = -20

        # elif kmh < 30:
        #     done = True

        else:
            done = False
            self.reward = +5

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            self.reward =1000
            done = True
            print("Max time elapsed")
        return self.front_Camera, self.reward, done, collided




def transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (224, 224))
    normalized = np.float32(resized) / 255.0
    return normalized

# class DQNCNN(torch.nn.Module):
#     def __init__(self, num_actions):
#         super(DQNCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
#         self.fc4 = nn.Linear(in_features=7*7*64, out_features=512)
#         self.fc5 = nn.Linear(in_features=512, out_features=num_actions)

#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.relu(self.conv3(x))
#         x = nn.functional.relu(self.fc4(x.view(x.size(0), -1)))
#         x = self.fc5(x)
#         return x
    

#     def act(self,obs,epsilon):
#         if np.random.rand()< epsilon:
#             return np.random.randint(4)
#         obs_t = torch.as_tensor(obs,dtype =torch.float32)
#         q_values = obs_t
#         max_q_index = torch.argmax(q_values,dim=1)
#         action = max_q_index.item()
#         return action

class DQNCNN(torch.nn.Module):
    def __init__(self, num_actions):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(in_features=24*24*64, out_features=512)  # Corrected input size
        self.fc5 = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)
        return x


    def act(self,obs,epsilon):
        if np.random.rand()< epsilon:
            return np.random.randint(4)
        obs_t = torch.as_tensor(obs,dtype =torch.float32)
        q_values = obs_t
        max_q_index = torch.argmax(q_values,dim=1)
        action = max_q_index.item()
        return action



env = CarEnv()
Gamma = 0.99
Batch_Size = 32
Buffer_Size =5000000
Min_Replay_Size = 10000
# random.seed(1)
# np.random.seed(1)
replay_buffer = deque(maxlen= Buffer_Size)
rew_buffer = deque([0.0],maxlen=100)
episode_reward = 0.0
online_net = DQNCNN(6)
target_net = DQNCNN(6)
losses =[]
obses = []
actions =[]
rewards =[]
dones = []
new_obses =[]
obses_i =0
actions_i=0
rewards_i=0
dones_i=0
new_obses_i=0
episodes = 500
success = 0
total_rewards_CARLA_DDQN =[]
eps =[]
batch_size = 32
epsilon = 1.0    
min_epsilon = 0.01 
tau = 0.999
decay= (min_epsilon/epsilon)**(1/episodes)
target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(),lr = 5e-4)
     
obs = env.reset()
obs = transform(obs)
obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
for _ in range(Min_Replay_Size):
    #obs = env.reset()
    action = env.action_space.sample()
    new_obs,reward, done, collided = env.step(action)
    new_obs = transform(new_obs)
    new_obs = torch.tensor(new_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    transition = (obs, action,reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    if done:
        print("done")
        for actor in env.actor_list:
            actor.destroy()
            env.actor_list =[]
            time.sleep(1)     
        env.reset()
        # env.reset()   
for actor in env.actor_list:
    actor.destroy()
    env.actor_list =[]
    time.sleep(1)   
for episode in range(episodes):
    print("Episode:", episode)
    obs = env.reset()
    obs = transform(obs)
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    done = False
    truncate = False
    success = 0
    losses_ = 0
    timestep =0
    episode_start = time.time()
    while not done:
        rnd_sample = random.uniform(0, 1)
        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = online_net(obs).squeeze().numpy()
                action = np.argmax(q_values)
        new_obs, reward, done, info = env.step(action)
        new_obs = transform(new_obs)
        new_obs = torch.tensor(new_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        transition = (obs, action, reward, done, new_obs)
        replay_buffer.append(transition)
        obs = new_obs
        success += reward
        if len(replay_buffer) > batch_size:
            transitions = random.sample(replay_buffer, batch_size)
            obses = torch.cat([t[0] for t in transitions])
            actions = torch.as_tensor([t[1] for t in transitions], dtype=torch.int64)
            rewards = torch.as_tensor([t[2] for t in transitions], dtype=torch.float32)
            dones = torch.as_tensor([t[3] for t in transitions], dtype=torch.float32)
            new_obses = torch.cat([t[4] for t in transitions])
            obses_t = obses
            actions_t = actions.unsqueeze(-1)
            rewards_t = rewards.unsqueeze(-1)
            dones_t = dones.unsqueeze(-1)
            new_obses_t = new_obses
            q_values = online_net(obses_t)
            q_values = q_values.gather(1,actions_t)
            next_q_values = target_net(new_obses_t)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            nexta_q_values = next_q_values.max(1)[0].unsqueeze(-1)
            expected_q_values = rewards_t + Gamma*nexta_q_values*(1-dones_t)
            loss = F.mse_loss(q_values,expected_q_values)
            # target_q_values = target_net(new_obses_t)
            # max_target_q_values = target_q_values.max(dim =1,keepdim = True)[0]
            # targets = rewards_t + Gamma*(1- dones_t)* max_target_q_values
            # q_values = online_net(obses_t)
            # action_q_values = torch.gather(input= q_values,dim =1, index = actions_t)
            # loss = F.mse_loss(action_q_values,targets)
            # losses_ = loss.item()
            # print("loss:",loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            timestep+= 1
            current_time = time.time()
            if (current_time -episode_start) >30:
                done = True
                success = 2000
                print("max time")
    time_taken = current_time - episode_start
    for actor in env.actor_list:
            actor.destroy()
            env.actor_list =[]  
    eps.append(epsilon)
    total_rewards_CARLA_DDQN.append(success)
    losses.append(losses_)
    epsilon = epsilon*decay
    print(f"Episode {episode} and reward is {success} timestep: {timestep} time is: {time_taken}")
    if (episode %100 ==0):
        print(f"Episode: {episode} and reward: {np.mean(total_rewards_CARLA_DDQN[-100:])}")#.format(episode,))
    if (episode % 4 ==0):
        target_net.load_state_dict(online_net.state_dict())
    if (episode % 4 ==0):
        torch.save(online_net.state_dict(), 'rkhatik_final_project_ddqn_carla.pth')
    if (episode % 1 ==0):
        for target_param, param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
torch.save(online_net.state_dict(), 'rkhatik_final_project_ddqn_carla.pth')



plt.figure(figsize=(15, 10))
plt.plot(total_rewards_CARLA_DDQN, 'r')
plt.xlabel('Episode', fontsize=28)
plt.ylabel('Reward Value', fontsize=28)
plt.title('Rewards Per Episode for CARLA DDQN', fontsize=36)
plt.ylim(ymin=min(total_rewards_CARLA_DDQN), ymax=max(total_rewards_CARLA_DDQN)+1)
plt.xlim(xmin=0, xmax= episodes)
plt.grid()
plt.savefig('rewards_DDQN.png')
plt.show()


#plt.savefig('foo.png')

##########################################################################

plt.figure(figsize=(15, 10))
plt.plot(eps, linewidth=4)
plt.xlabel('Episode', fontsize=28)
plt.ylabel('Epsilon Values', fontsize=28)
plt.title('Epsilon Decay', fontsize=36)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid()
plt.savefig('epsilon_DDQN.png')
plt.show()


print("done")
