import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import math
#from google.colab import files 
import environment_simulation_move as env
from ReplayBuffer import ReplayBuffer,create_directory
from TD3_agent import TD3
import random
print(T.__version__)

for i_loop in range(6):
    #can only deal with 10 users per ap at most
    numAPuser = 5
    numRU = 8
    # numSenario = 4
    linkmode = 'uplink'
    ru_mode = 3
    episode = 10000
    max_iteration = 200
    test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)

    TD3_agent_s = TD3(alpha=1e-4, beta=2e-4,numSenario=1,numAPuser=numAPuser,numRU=numRU,
        actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
        actor_fc4_dim=2**6,
        critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
        critic_fc4_dim=2**6,
        ckpt_dir='./TD3_s/',
        gamma=0.99,tau=0.001,action_noise=1e-5,max_size=100000,batch_size=128)
    create_directory('./TD3_s/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])

    TD3_agent_m = TD3(alpha=1e-4, beta=2e-4,numSenario=3,numAPuser=numAPuser,numRU=numRU,
        actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
        actor_fc4_dim=2**6,
        critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
        critic_fc4_dim=2**6,
        ckpt_dir='./TD3_m/',
        gamma=0.99,tau=0.001,action_noise=1e-5,max_size=100000,batch_size=128)
    create_directory('./TD3_m/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])


    reward_history = []
    system_bitrate_history = []
    reward_ave_history = []
    system_ave_bitrate_history = []
    
    for i_episode in range(episode):
        actor_loss_history = []
        critic_loss_history = []
        test_env.change_RU_mode(4)
        x_init,y_init = test_env.senario_user_local_init()
        x,y = x_init,y_init
        userinfo = test_env.senario_user_info(x,y)
        channel_gain_obs = test_env.channel_gain_calculate()
        observation = test_env.get_sinr()
        test_env.change_RU_mode(3)
