import torch as T
import torch.nn.functional as F
import numpy as np
from TD3_network import ActorNetwork, CriticNetwork
from ReplayBuffer import ReplayBuffer

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class TD3:
    def __init__(self, alpha, beta, numSenario, numAPuser, numRU, 
                 actor_fc1_dim, actor_fc2_dim, actor_fc3_dim, actor_fc4_dim, 
                 critic_fc1_dim, critic_fc2_dim, critic_fc3_dim, critic_fc4_dim,  
                 ckpt_dir,
                 gamma=0.99, tau=0.005, action_noise=0.1, max_size=1000000,
                 batch_size=128, update_actor_interval=10):    
                 
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.checkpoint_dir = ckpt_dir
        self.update_actor_interval = update_actor_interval
        self.total_train = 0

        self.actor = ActorNetwork(alpha=alpha, state_dim=numSenario*numAPuser*numRU, action_dim=numAPuser*numRU,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim,
                                  fc3_dim=actor_fc3_dim, fc4_dim=actor_fc4_dim)   

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=numSenario*numAPuser*numRU, action_dim=numAPuser*numRU,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim,
                                         fc3_dim=actor_fc3_dim, fc4_dim=actor_fc4_dim)

        self.critic = CriticNetwork(beta=beta, state_dim=numSenario*numAPuser*numRU, action_dim=numAPuser*numRU,
                                    q1_fc1_dim=critic_fc1_dim, q1_fc2_dim=critic_fc2_dim,
                                    q1_fc3_dim=critic_fc3_dim, q1_fc4_dim=critic_fc4_dim,
                                    q2_fc1_dim=critic_fc1_dim, q2_fc2_dim=critic_fc2_dim,
                                    q2_fc3_dim=critic_fc3_dim, q2_fc4_dim=critic_fc4_dim)

        self.target_critic = CriticNetwork(beta=beta, state_dim=numSenario*numAPuser*numRU, action_dim=numAPuser*numRU,
                                           q1_fc1_dim=critic_fc1_dim, q1_fc2_dim=critic_fc2_dim,
                                           q1_fc3_dim=critic_fc3_dim, q1_fc4_dim=critic_fc4_dim,
                                           q2_fc1_dim=critic_fc1_dim, q2_fc2_dim=critic_fc2_dim,
                                           q2_fc3_dim=critic_fc3_dim, q2_fc4_dim=critic_fc4_dim) 
                                            
        self.memory = ReplayBuffer(max_size=max_size, numSenario=numSenario,
                                   numAPuser=numAPuser, numRU=numRU,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=1.0)          
        #### 展示用
        self.critic_loss_show = 0
        self.actor_loss_show = 0


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(),
                                                       self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)
        

    def choose_action(self, observation, train=True):
        self.actor.eval()
        state = T.tensor(np.array([observation]), dtype=T.float).to(device) ## 在外部转化为张量
        action = self.actor.forward(state).squeeze() ## squeeze减少维度

        if train:
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)
            action = T.clamp(action+noise, 1e-15, 1) ###1e-15可能更好
        self.actor.train()

        return action.detach().cpu().numpy()

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)


    def learn(self):
        if not self.memory.ready():
            return

        states, actions, reward, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(reward, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)
        self.total_train += 1

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)

            q1_, q2_= self.target_critic.forward(next_states_tensor, next_actions_tensor)
            q1_ = T.squeeze(q1_)
            q2_ = T.squeeze(q2_)            
            q_ = T.min(q1_, q2_)
            target = rewards_tensor + self.gamma * q_

        q1, q2 = self.critic.forward(states_tensor, actions_tensor)
        q1 = T.squeeze(q1)
        q2 = T.squeeze(q2)        
        critic_loss = F.mse_loss(q1, target.detach()) + F.mse_loss(q2, target.detach())

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # 延迟目标网络更新
        if self.total_train % self.update_actor_interval == 0:
            new_actions_tensor = self.actor.forward(states_tensor)
            actor_loss = -T.mean(self.critic.AQ(states_tensor, new_actions_tensor))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.critic_loss_show = critic_loss.detach().cpu().numpy() ##############################
            self.actor_loss_show = actor_loss.detach().cpu().numpy() ################################

            self.update_network_parameters()


    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic.save_checkpoint(self.checkpoint_dir + 'Critic/TD3_critic_{}'.format(episode))
        print('Saving critic network successfully!')
        self.target_critic.save_checkpoint(self.checkpoint_dir +
                                           'Target_critic/TD3_target_critic_{}'.format(episode))
        print('Saving target critic network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic.load_checkpoint(self.checkpoint_dir + 'Critic/TD3_critic_{}'.format(episode))
        print('Loading critic network successfully!')
        self.target_critic.load_checkpoint(self.checkpoint_dir +
                                           'Target_critic/TD3_target_critic_{}'.format(episode))
        print('Loading target critic network successfully!')