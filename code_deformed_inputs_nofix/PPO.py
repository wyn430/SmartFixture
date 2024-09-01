import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from Shared_Net import Shared_Module, Shared_Module_GCN, Actor1_Arm, Actor2_Arm, Critic_Arm, Actor2_Arm_Conv, Critic_Arm_new

torch.manual_seed(24)


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions2 = []
        self.states = []
        self.logprobs2 = []
        self.rewards = []
        self.is_terminals = []
        self.mask = []
    

    def clear(self):
        del self.actions2[:]
        del self.states[:]
        del self.logprobs2[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.mask[:]


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
        action2_probs = torch.nan_to_num(action2_probs, nan=1/3901, posinf=1/3901, neginf=1/3901)
        dist2 = Categorical(action2_probs)
        action2 = dist2.sample()
        action2_logprob = dist2.log_prob(action2)

        
        
        return action2.detach(), action2_logprob.detach()
    

    def evaluate(self, x, state, action2, mask):

        feature = self.shared_net.forward(x, state)
        
        
        action2_probs = self.actor2.forward(feature, mask)
        try:
            action2_probs = torch.nan_to_num(action2_probs, nan=1/3901, posinf=1/3901, neginf=1/3901)
            dist2 = Categorical(action2_probs)
        except:
            action2_probs = torch.nan_to_num(action2_probs, nan=1/3901, posinf=1/3901, neginf=1/3901)
            sum_action2_probs = torch.sum(action2_probs, dim = 1)
            zeros_ind = (sum_action2_probs==0).nonzero()[:,0]
            action2_probs[zeros_ind] = 1/3901
            dist2 = Categorical(action2_probs)
        action2_logprob = dist2.log_prob(action2)
        dist2_entropy = dist2.entropy()
        
        state_values = self.critic(feature)
        
        return state_values, action2_logprob, dist2_entropy


class PPO:
    def __init__(self, state_dim, action2_dim, shared_channel_list, actor_arm_dim_list, critic_arm_dim_list, emb_dims, feature_dims, k, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device, checkpoint_path = None):
        
        

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action2_dim, 
                                  shared_channel_list, actor_arm_dim_list, critic_arm_dim_list, 
                                  emb_dims, feature_dims, k, self.device).to(self.device)
        
        if checkpoint_path:
            if torch.cuda.device_count() == 0:
                self.policy.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
            elif torch.cuda.device_count() > 0:
                self.policy.load_state_dict(torch.load(checkpoint_path))
            #self.policy.load_state_dict(torch.load(checkpoint_path))
            print('loaded')
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor2.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        print("Let's use", torch.cuda.device_count(), "GPUs!")
      

        self.policy_old = ActorCritic(state_dim, action2_dim, 
                                  shared_channel_list, actor_arm_dim_list, critic_arm_dim_list, 
                                  emb_dims, feature_dims, k, self.device).to(self.device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def select_action(self, x, state, mask):

        self.policy_old.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            x = torch.FloatTensor(x).to(self.device)
            mask = torch.FloatTensor(mask).to(self.device)
            
            action2, action2_logprob = self.policy_old.act(x, state, mask)

        self.buffer.states.append(state)
        self.buffer.actions2.append(action2)
        self.buffer.logprobs2.append(action2_logprob)
        self.buffer.mask.append(mask)

        return action2.item()


    def update(self, x, train_batch):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        self.policy.train()
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions2 = torch.squeeze(torch.stack(self.buffer.actions2, dim=0)).detach().to(self.device)
        old_logprobs2 = torch.squeeze(torch.stack(self.buffer.logprobs2, dim=0)).detach().to(self.device)
        old_masks = torch.squeeze(torch.stack(self.buffer.mask, dim=0)).detach().to(self.device)
        
        sample_size = old_states.size(0)
        
        x = torch.FloatTensor(x).to(self.device)
        x = x.repeat(sample_size, 1, 1)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            
            for i in range(int(sample_size / train_batch)):
                start_ind = i*train_batch
                

                # Evaluating old actions and values
                state_values, logprobs2, dist2_entropy = self.policy.evaluate(
                    x[start_ind:start_ind+train_batch],
                    old_states[start_ind:start_ind+train_batch],
                    old_actions2[start_ind:start_ind+train_batch],
                    old_masks[start_ind:start_ind+train_batch])

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios2 = torch.exp(logprobs2 - old_logprobs2[start_ind:start_ind+train_batch].detach())

                # Finding Surrogate Loss
                advantages = rewards[start_ind:start_ind+train_batch] - state_values.detach()   
                surr1 = ratios2 * advantages
                surr2 = torch.clamp(ratios2, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards[start_ind:start_ind+train_batch]) - 0.01 * dist2_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


