import os
import glob
import time
import sys
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl

import torch
import numpy as np
import random

from Ansys_env import Env

from PPO import PPO



################################### Training ###################################

def train(pre_train,pre_train_dir,checkname,ip_addr):
#def train():

    print("============================================================================================")
    
    
    ################################## set device ##################################

    # set device to cpu or cuda
    device = torch.device('cpu')

    if(torch.cuda.is_available()): 
        device = torch.device('cuda') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")

    ####### initialize environment hyperparameters ######

    env_name = "Ansys_assembly"

    quant_lim = 8                      # maximum number of fixtures
    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(30001)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(max_ep_len)          # save model frequency (in num timesteps)
    
    thresh = 0.45 ##controls the minimum distance between two fixtures
    
    original_input_filename = 'input_nondeform_2mm.inp'
    initial_fixture_locations = [2552, 2578, 2628]
    max_num = 20
    ip = ip_addr
    port = 8800
    
    print("training environment name : " + env_name)
    

    #####################################################


    ## Note : print/log frequencies should be > than max_ep_len


    ################ PPO hyperparameters ################

    update_timestep = max_ep_len      # update policy every n timesteps
    K_epochs = 30               # update policy for K epochs in one PPO update
    train_batch = 20
    
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)

    #####################################################
    
    ############### Network Hyperparameters #############
    
    shared_channel_list = [1, 8, 16, 16] #this may not be used when the shared module is GCN
    #actor_arm_dim_list = [256, 128, 64]
    actor_arm_dim_list = [1024, 512, 256]
    critic_arm_dim_list = [512, 128, 64]
    emb_dims = 512
    feature_dims = 6
    k = 8


    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    
    if pre_train == 1:
        timestamp = pre_train_dir
        print('continue the training for: ' + timestamp)
    else:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    '''
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    '''
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/' + timestamp + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    
    input_filename = log_dir + original_input_filename
    copyfile(original_input_filename, input_filename)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)


    #### create new log file for each run
    log_f_name = log_dir + 'PPO_' + env_name + "_log_" + str(run_num) + ".txt"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    ###################### Initialize Environment ######################
    
    print("initialize environment")
    
    #mapdl = Mapdl(ip=ip, port=port, request_instance=False)
    env = Env(original_input_filename, input_filename, initial_fixture_locations, max_ep_len, thresh, ip, port)

    # state space dimension
    state_dim = env.get_state_shape()

    # action space dimension
    
    action2_dim = env.get_action_shape()

    
    
    #####################################################
    
    
    ################### checkpointing ###################

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/' + timestamp + '/'
    
    if not os.path.exists(directory):
          os.makedirs(directory)
    
    current_num_check = 0
    '''
    checkpoint_path = None
    '''
    if pre_train == 1:
        current_num_check = len(next(os.walk(directory))[2])
        checkpoint_name = checkname
        checkpoint_path = directory + checkpoint_name
        print("load the model from : " + checkpoint_path)
        
    else:
        checkpoint_path = None
    
    
    

    #####################################################


    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action2 space dimension : ", action2_dim)

    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action2_dim, shared_channel_list, actor_arm_dim_list, critic_arm_dim_list, 
                    emb_dims, feature_dims, k, lr_actor, lr_critic, gamma, K_epochs, 
                    eps_clip, device, checkpoint_path = checkpoint_path)


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")


    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0


    # training loop
    while time_step <= max_training_timesteps:
        
        
        x, state, mask = env.reset(initial_fixture_locations, original_input_filename, input_filename)
        current_ep_reward = 0

        #for t in tqdm(range(1, max_ep_len+1)):
        for t in tqdm(range(1, max_ep_len+1)):

            # select action with policy
            action2 = ppo_agent.select_action(x, state, mask)
            
            x, state, reward, done, mask = env.step(action2, original_input_filename, input_filename, quant_lim)
            
            #print(done)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            

            time_step +=1
            current_ep_reward += reward
            
            
            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
                
            # update PPO agent
            if time_step % update_timestep == 0:
                print("start updating")
                ppo_agent.update(x, train_batch)

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                checkpoint_path = directory + "PPO_{}_{}_{}_{}.pth".format(env_name, random_seed, run_num, str(int(time_step/save_model_freq) + current_num_check))
                print("save checkpoint path : " + checkpoint_path)
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                print('done')
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    _, _, _ = env.reset(initial_fixture_locations, original_input_filename, input_filename)
    log_f.close()
    




    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")




if __name__ == '__main__':
    
    pre_train = int(sys.argv[1])
    if pre_train == 0:
        pre_train_dir = None
        checkname = None
    elif pre_train == 1:
        pre_train_dir = str(sys.argv[2])
        checkname = str(sys.argv[3])
    ip_addr = str(sys.argv[4])
    train(pre_train,pre_train_dir,checkname,ip_addr)
    #train()
    
    
    
    
    
    
    
