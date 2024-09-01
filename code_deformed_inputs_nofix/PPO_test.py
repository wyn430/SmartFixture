import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from Shared_Net import Shared_Module, Shared_Module_GCN, Actor1_Arm, Actor2_Arm, Critic_Arm, Actor2_Arm_Conv, Critic_Arm_new




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        
        self.actions2 = []
        self.states = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        
        del self.actions2[:]
        del self.states[:]
        del self.rewards[:]
        del self.is_terminals[:]




class PPO:
    def __init__(self, state_dim, action2_dim, shared_channel_list, actor_arm_dim_list, critic_arm_dim_list, emb_dims, 
                 feature_dims, k, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device, check_point):
        
        

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.buffer = RolloutBuffer()
 
        print("Let's use", torch.cuda.device_count(), "GPUs!")
      

        self.policy_old = ActorCritic(state_dim, action2_dim, 
                                      shared_channel_list, actor_arm_dim_list, critic_arm_dim_list, 
                                      emb_dims, feature_dims, k, self.device).to(self.device)
        
        if torch.cuda.device_count() == 0:
            self.policy_old.load_state_dict(torch.load(check_point, map_location=torch.device('cpu')))
        elif torch.cuda.device_count() > 0:
            self.policy_old.load_state_dict(torch.load(check_point))


    def select_action(self, x, state, mask):

        self.policy_old.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            x = torch.FloatTensor(x).to(self.device)
            mask = torch.FloatTensor(mask).to(self.device)
            
            action2 = self.policy_old.act_test(x, state, mask)

        self.buffer.states.append(state)
        
        self.buffer.actions2.append(action2)

        return action2.item()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action2_dim, shared_channel_list, actor_arm_dim_list, critic_arm_dim_list, emb_dims, feature_dims, k, device):
        super(ActorCritic, self).__init__()
        
        ##shared_net is a stack of conv layers shared by the actor and critic, channel_list represents the list of input and output channels
        ##e.g. shared_channel_list = [1, output1, output2, ....]
        ##arm_dim_list = [H * W * channel_list[-1], output1, output2, ...]
        ##action1_dim = 2, add fixture or not
        ##action2_dim = H*W, position
        
        
        #self.shared_net = Shared_Module(shared_channel_list)
        #H,W = state_dim
        #shared_output_dim = H * W * channel_list[-1]
        
        self.shared_net = Shared_Module_GCN(feature_dims, k, emb_dims, device)      
        
        self.actor2 = Actor2_Arm(emb_dims*2, actor_arm_dim_list, action2_dim)
        self.critic = Critic_Arm(emb_dims*2, critic_arm_dim_list)
        
        self.shared_net = nn.DataParallel(self.shared_net)
        
        self.actor2 = nn.DataParallel(self.actor2)
        self.critic = nn.DataParallel(self.critic)


    def forward(self):
        raise NotImplementedError
    

    def act(self, x, state, mask):

        feature = self.shared_net.forward(x, state)
        
        
        action2_probs = self.actor2.forward(feature, mask)
        dist2 = Categorical(action2_probs)
        action2 = dist2.sample()
        action2_logprob = dist2.log_prob(action2)

        
        
        return action2.detach(), action2_logprob.detach()
    
    def act_test(self, x, state, mask):

        feature = self.shared_net.forward(x, state)
        
        
        
        action2_probs = self.actor2.forward(feature, mask)
        
        action2 = torch.argmax(action2_probs)
        
        #print(action1_probs)
        #print(action2_probs.max())

        return action2.detach()
    

    def evaluate(self, x, state, action1, action2, mask):

        feature = self.shared_net.forward(x, state)
        
        
        action2_probs = self.actor2.forward(feature, mask)
        dist2 = Categorical(action2_probs)
        action2_logprob = dist2.log_prob(action2)
        dist2_entropy = dist2.entropy()
        
        state_values = self.critic(feature)
        
        return state_values, action2_logprob, dist2_entropy


