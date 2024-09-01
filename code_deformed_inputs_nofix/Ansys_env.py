import numpy as np
import random
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
import os
import re
import time
from timeout import timeout
from scipy.spatial import distance
random.seed(0)
np.random.seed(0)



class Env:
    
    def __init__(self, original_input_filename, input_filename, initial_fixture_locations, max_num, thresh, ip, port, initial_deform):
        
        self.ip = ip
        self.port = port
        self.mapdl = Mapdl(ip=ip, port=port, request_instance=False)
        self.initial_fixture_locations = initial_fixture_locations.copy()
        self.previous_fixtures = initial_fixture_locations.copy()
        #self.new_fixtures = initial_fixture_locations.copy()
        self.new_fixtures = []
        self.indexed_fixture = []
        self.max_num = max_num
        self.num_steps = 0
        self.thresh = thresh
        self.bottom_index = None
        
        self.bottom_index, self.bottom_surface_nodes, self.bottom_displacement, self.bottom_stress = self.get_results(input_filename, initial_deform)
        #print(self.bottom_index.shape)
        
        self.state = np.copy(self.bottom_displacement)
        #self.state = np.concatenate((self.bottom_displacement, self.bottom_surface_nodes), axis = 1)
        self.mask = self.get_mask()
        
    def reset(self, initial_fixture_locations, original_input_filename, input_filename, initial_deform):
        
        self.num_steps = 0
        ###use the new fixtures to first reset the input filename to starting point
        self.new_fixtures = []
        self.initial_fixture_locations = initial_fixture_locations.copy()
        #print(initial_fixture_locations)
        self.update_inputfile(original_input_filename, input_filename)
        
        #self.new_fixtures = []
        self.previous_fixtures = initial_fixture_locations.copy()
            
        _, self.bottom_surface_nodes, self.bottom_displacement, self.bottom_stress = self.get_results(input_filename, initial_deform)
        
        self.state = np.copy(self.bottom_displacement)
        #self.state = np.concatenate((self.bottom_displacement, self.bottom_surface_nodes), axis = 1)
        self.mask = self.get_mask()
        
        return np.expand_dims(self.bottom_surface_nodes, axis=0), np.expand_dims(self.state, axis=0), np.expand_dims(self.mask, axis=0)
      
        
        
    def step(self, action2, original_input_filename, input_filename, quant_lim, initial_deform):

        ##one more fixture is added
        ## note that self.new_fixtures stores the index of fixtures in Ansys, which starts from 1, the index of fixture in self.bottom_index starts from 0
        fixture_index = self.bottom_index[action2] + 1
        self.new_fixtures.append(fixture_index)
        #print(action2)
        #print(self.bottom_index[action2])
        #print(fixture_index)
        
        #if len(self.new_fixtures) >= 3:
            #self.update_inputfile(original_input_filename, input_filename)
        if len(self.new_fixtures) >= 3:
            self.update_inputfile(original_input_filename, input_filename)    
            self.previous_fixtures = self.new_fixtures.copy()

            ###Solve the FEA with the latest input
        _, self.bottom_surface_nodes, self.bottom_displacement, self.bottom_stress = self.get_results(input_filename, initial_deform)
        #print(self.bottom_displacement.max())

        reward = self.get_reward()
        self.state = np.copy(self.bottom_displacement)
        #self.state = np.concatenate((self.bottom_displacement, self.bottom_surface_nodes), axis = 1)
        self.mask = self.get_mask()
            
            
        
        self.num_steps += 1 
        done = self.terminate(quant_lim)
        
        return np.expand_dims(self.bottom_surface_nodes, axis=0), np.expand_dims(self.state, axis=0), reward, done, np.expand_dims(self.mask, axis=0)
    
    @timeout(30)
    def remote_run(self, input_filename, initial_deform):
        
        
        self.mapdl.clear()
        self.mapdl.input(input_filename)
        result = self.mapdl.result
        
        
        if self.bottom_index is None:
            bottom_index = np.where(self.mapdl.mesh.nodes[:,1] == 0)[0]
            
        else:
            bottom_index = None
            
        initial_deform_copy = initial_deform.copy()
        #print(bottom_surface_nodes.shape)
        _, displacement =  result.nodal_displacement(0, in_nodal_coord_sys = True)
        #print(displacement.shape)
        displacement += initial_deform_copy
        
        _, stress = result.principal_nodal_stress(0)
        
        if bottom_index is not None:
            bottom_surface_nodes = self.mapdl.mesh.nodes[bottom_index]
            bottom_displacement = displacement[bottom_index]
            bottom_stress = stress[bottom_index][:,:3]
            if len(self.new_fixtures) < 3:
                bottom_displacement = initial_deform_copy[bottom_index]
                bottom_stress[:,:] = 0
        else:
            bottom_surface_nodes = self.mapdl.mesh.nodes[self.bottom_index]
            bottom_displacement = displacement[self.bottom_index]
            bottom_stress = stress[self.bottom_index][:,:3]
            if len(self.new_fixtures) < 3:
                bottom_displacement = initial_deform_copy[self.bottom_index]
                bottom_stress[:,:] = 0

        bottom_stress = self.preprocess(bottom_stress)
        #print(bottom_displacement.max())
        self.mapdl.finish()
        
        
            
        '''    
        else:
            self.mapdl.clear()
            self.mapdl.input(input_filename)
            result = self.mapdl.result

            bottom_index = np.where(self.mapdl.mesh.nodes[:,1] == 0)[0]
            bottom_surface_nodes = self.mapdl.mesh.nodes[bottom_index]

            _, displacement =  result.nodal_displacement(0, in_nodal_coord_sys = True)
            bottom_displacement = displacement[bottom_index]
            bottom_displacement[:,:] = 0

            _, stress = result.principal_nodal_stress(0)
            bottom_stress = stress[bottom_index][:,:3]

            bottom_stress = self.preprocess(bottom_stress)
            bottom_stress[:,:] = 0
            #print(bottom_displacement.max())
            self.mapdl.finish()
        '''
        
        return bottom_index, bottom_surface_nodes, bottom_displacement, bottom_stress
        
    def get_results(self, input_filename, initial_deform):
        
        try:
            bottom_index, bottom_surface_nodes, bottom_displacement, bottom_stress = self.remote_run(input_filename, initial_deform)
                       
        except:
            print("exit ansys and try to reconnect 5 times")
            
            try:
                self.mapdl.exit()
                print("remote exit")
                time.sleep(30)
            except:
                time.sleep(30)
                
            i = 0
            while i <= 5:
                i += 1
                try: 
                    self.mapdl = Mapdl(ip=self.ip, port=self.port, request_instance=False)
                    print("sucessfully reconnect")
                    bottom_index, bottom_surface_nodes, bottom_displacement, bottom_stress = self.remote_run(input_filename, initial_deform)
            
                    break
                except:
                    try:
                        print("reconnect fails and remote exit again")
                        self.mapdl.exit()
                        time.sleep(30)
                    except:
                        time.sleep(30)
        
        return bottom_index, bottom_surface_nodes, bottom_displacement, bottom_stress
    
    
    def get_reward(self):
        
        ## the goal of this reward, encourage improvement, and encourage large improvement
        reward = -1
            
        try:
            previous_max_displacement = np.linalg.norm(self.state[:,:3], axis = 1).max()
            current_max_displacement = np.linalg.norm(self.bottom_displacement, axis = 1).max()
            delta_displacement = current_max_displacement - previous_max_displacement
        except:
            print('error in getting reward')
            previous_max_displacement = 0
            current_max_displacement = 10
            delta_displacement = 0
            
        #print(current_max_displacement)
        
        if len(self.new_fixtures) == 8:
            reward += 1
            reward += np.log10(1/current_max_displacement) * 0.1
        else:
            reward += 1
            
        '''        
        if delta_displacement < 0:
            reward += 1
            
            
            reward += np.log10(1/current_max_displacement) * 0.1
            
         
        ##otherwise the fixtures decay the performance will be punished
        elif delta_displacement > 0:
            reward += 0
        '''
        
        return reward
    
    def terminate(self, quant_lim):
        
        if self.num_steps >= self.max_num or len(self.new_fixtures) >= quant_lim:
            return True
        
        else:
            return False
            
    
    def update_inputfile(self, original_input_filename, input_filename):
        
        spaces = '      '

        ###modify the input file to update the location of fixtures
        #previous_fix_str = spaces + spaces.join([str(elem) for elem in self.previous_fixtures])
        new_fix_str = spaces + spaces.join([str(elem) for elem in self.new_fixtures])
        previous_fix_str = spaces + spaces.join([str(elem) for elem in self.initial_fixture_locations])
        
        with open(original_input_filename, 'r') as f:
            text = f.read()
            text = re.sub('CMBLOCK,_FIXEDSU,NODE,\s+\d+', 'CMBLOCK,_FIXEDSU,NODE, ' + str(len(self.new_fixtures)), text)
            #print(previous_fix_str)
            #print(new_fix_str)
            text = re.sub(previous_fix_str, new_fix_str, text)
            f.seek(0)
            #f.write(text)
            f.close()
            
        with open(input_filename, 'w') as f_new:
            f_new.write(text)
            f_new.close()
            
    def get_mask(self):
        
        ## note that self.new_fixtures stores the index of fixtures in Ansys, which starts from 1, the index of fixture in self.bottom_index starts from 0
        try:
            self.indexed_fixture = self.new_fixtures.copy()
            self.indexed_fixture = [elem - 1 for elem in self.indexed_fixture]
            actions = np.where(np.isin(self.bottom_index, self.indexed_fixture))
            #print(actions)
            bottom_nodes = np.copy(self.bottom_surface_nodes) # N * 3

            actions_position = bottom_nodes[actions] # n * 3
            distances = distance.cdist(bottom_nodes, actions_position, 'euclidean') # N * n
            min_distance = np.min(distances, axis = 1)
            #print(min_distance.max())
            #print(min_distance.mean())
            #print(zero_positions)
            mask = np.ones(bottom_nodes.shape[0])
            mask[min_distance <= self.thresh] = 0
            #print(mask.sum())
            #print(mask[actions])
        
        
        except:
            self.indexed_fixture = self.new_fixtures.copy()
            self.indexed_fixture = [elem - 1 for elem in self.indexed_fixture]
            zero_positions = np.where(np.isin(self.bottom_index, self.indexed_fixture))
            #print(zero_positions)
            mask = np.ones(self.bottom_surface_nodes.shape[0])
            mask[zero_positions] = 0
        
        
        return mask
        
    def get_state_shape(self):
        
        return self.bottom_surface_nodes.shape
        
        
    def get_action_shape(self):
        
        return self.bottom_surface_nodes.shape[0]
    
    def preprocess(self, data):
        ##replace the nan in stress and rescale the stress
        
        return np.nan_to_num(data) / 100000000
    
    def get_reward_old(self, action1):
        
        ##under this reward, the maximum culmulative reward appears at when the RL-agent select the maximum number of fixtures and ensure add each of them will have a subtle improvement.
        reward = 0
        
        if action1 == 0:
            reward -= 1
            
            delta_displacement = np.linalg.norm(self.bottom_displacement, axis = 1).max() - np.linalg.norm(self.state[:,:3], axis = 1).max()
            
            delta_stress = np.linalg.norm(self.bottom_stress, axis = 1).max() - np.linalg.norm(self.state[:,3:6], axis = 1).max()
            
            if delta_displacement and delta_stress < 0:
                reward += 2
                
            elif delta_displacement or delta_stress < 0:
                reward += 1.5
                
            else:
                reward += 0
                
            
            #print(reward)    
        elif action1 == 1:
            reward += 0
        
        
        return reward
    
        
        
        
        