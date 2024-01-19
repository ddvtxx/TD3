import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

## state_dim = 1*4*5*8 = 160
## action_dim = 1*5*8 = 40 
def weight_init(m):   ## 初始化权重
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, 
                 fc1_dim, fc2_dim, fc3_dim, fc4_dim):
        super(ActorNetwork, self).__init__()
        self.input_layer = nn.Flatten() #将矩阵展开  (bacthsize,4,5,8)展开为(bacthsize,160)
        self.fc1 = nn.Linear(state_dim, fc1_dim) ## 
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.fc4 = nn.Linear(fc3_dim, fc4_dim)

        
        self.action = nn.Linear(fc4_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.ln1(self.fc1(self.input_layer(state))))
        x = T.relu(self.ln2(self.fc2(x)))
        x = T.relu(self.fc3(x))
        x = T.relu(self.fc4(x))
        action = F.gumbel_softmax(self.action(x), tau=1, hard=False, eps = 1e-5) 
        # action = T.reshape(action,(state.shape[0],state.shape[2],state.shape[3]))
        return action

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, 
                 q1_fc1_dim, q1_fc2_dim, q1_fc3_dim, q1_fc4_dim,
                 q2_fc1_dim, q2_fc2_dim, q2_fc3_dim, q2_fc4_dim):
        super(CriticNetwork, self).__init__()
        self.input_layer = nn.Flatten(1,-1) #将矩阵展开
        # Q1 architecture
        self.q1_fc1 = nn.Linear(state_dim*2, q1_fc1_dim) # (1,4,5,8)展开为(1,160)
        self.q1_ln1 = nn.LayerNorm(q1_fc1_dim)
        self.q1_fc2 = nn.Linear(q1_fc1_dim, q1_fc2_dim)
        self.q1_ln2 = nn.LayerNorm(q1_fc2_dim)
        self.q1_fc3 = nn.Linear(q1_fc2_dim, q1_fc3_dim)
        self.q1_fc4 = nn.Linear(q1_fc3_dim, q1_fc4_dim)
        self.q1 = nn.Linear(q1_fc4_dim, 1)
        # Q2 architecture
        self.q2_fc1 = nn.Linear(state_dim*2, q2_fc1_dim) # (1,4,5,8)展开为(1,160)
        self.q2_ln1 = nn.LayerNorm(q2_fc1_dim)
        self.q2_fc2 = nn.Linear(q2_fc1_dim, q2_fc2_dim)
        self.q2_ln2 = nn.LayerNorm(q2_fc2_dim)
        self.q2_fc3 = nn.Linear(q2_fc2_dim, q2_fc3_dim)
        self.q2_fc4 = nn.Linear(q2_fc3_dim, q2_fc4_dim)
        self.q2 = nn.Linear(q2_fc4_dim, 1)        
        
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.001)
        self.apply(weight_init)
        self.to(device)


    def forward(self, state, action):  
######### action 与 state 组合以后同时输入网络 #########        
        state_shape = state.shape
        action = action.reshape((state_shape[0],state_shape[1],state_shape[2],state_shape[3]))
        state_action = T.cat((state,action),2)
        x = T.relu(self.q1_ln1(self.q1_fc1(self.input_layer(state_action))))
        x = T.relu(self.q1_ln2(self.q1_fc2(x)))
        x = T.relu(self.q1_fc3(x))
        x = T.relu(self.q1_fc4(x))
        q1 = self.q1(x)

        x = T.relu(self.q2_ln1(self.q2_fc1(self.input_layer(state_action))))
        x = T.relu(self.q2_ln2(self.q2_fc2(x)))
        x = T.relu(self.q2_fc3(x))
        x = T.relu(self.q2_fc4(x))
        q2 = self.q2(x)
        return q1,q2
        
    def AQ(self, state, action):
        state_shape = state.shape
        action = action.reshape((state_shape[0],state_shape[1],state_shape[2],state_shape[3]))
        state_action = T.cat((state,action),2)  
        x = T.relu(self.q1_ln1(self.q1_fc1(self.input_layer(state_action))))
        x = T.relu(self.q1_ln2(self.q1_fc2(x)))
        x = T.relu(self.q1_fc3(x))
        x = T.relu(self.q1_fc4(x))
        aq = self.q1(x)  
        return aq      

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))