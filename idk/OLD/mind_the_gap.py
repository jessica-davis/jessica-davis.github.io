import numpy as np
import networkx as nx
import sys
import os
from matplotlib import pyplot as plt
from math import atan2, cos, sin


class Group_Conversations:
    def __init__(self,network,model_params):
        self.network = network
        self.max_time = model_params['max_time']
        self.speaking_prob = model_params['speaking_prob'] #can use a lambda function if needed to be some function
        self.listening_prob = model_params['listening_prob']
        
        self.alpha = model_params['alpha'] #decay rate
        self.N = model_params['N']
        self.ground_truth = model_params['ground_truth']
        self.individual_information = dict()
        self.initial_state_function = model_params['initial_state']
        self.individual_list = list(range(model_params['N']))
        
    def initialize_groups(self):
        for i in range(self.N):
            self.individual_information[i] = Agent(i,self.initial_state_function(0))
            self.individual_information[i].neighbors = list(self.network.neighbors(i))
    
    def vector2angle(v):
        return atan2(v[1],v[0])
    
    def conversation_dynamics(self):
        
        for t in range(self.max_time):
            
            #step 1: who is speaking
            self.whos_speaking()
            
            #step 2: who is listening
            self.whos_listening()
            
            #step 3: interact
            updated_states = self.interact1D(t)
            
            #step 4: update and rest
            self.update_reset_group(updated_states)
            
            
    def whos_speaking(self):
        
        np.random.shuffle(self.individual_list)
        
        for i in self.individual_list:
            agent = self.individual_information[i]
            neighbors_speaking = sum([self.individual_information[neigh].currently_speaking for neigh in agent.neighbors])
            
            if np.random.random() < self.speaking_prob * np.exp(self.alpha*neighbors_speaking):
                agent.currently_speaking = 1
                agent.spoke.append(1)
            else:
                agent.spoke.append(0)
                
    def whos_listening(self):
        #if we want listening to be a function of something
        for i in self.individual_list:
            agent = self.individual_information[i]
            if np.random.random() < self.listening_prob:
                agent.currently_listening = 1
        pass
    
    def interact1D(self,t):
        updated_states = dict()
        for i,agent in self.individual_information.items():
       
            current_state = [agent.states[-1]]
            neighbor_influence = []
            bias =[np.random.normal(loc = self.ground_truth,scale =.1)]
            
            if agent.currently_listening:
                for neigh in agent.neighbors:       
                    neighboring_agent = self.individual_information[neigh]
                    if neighboring_agent.currently_speaking:
                        neighbor_influence.append(neighboring_agent.states[-1])
            updated_states[i] = np.mean(current_state+neighbor_influence+bias)
        return updated_states
     
    def interact2D(self,t):
        updated_states = dict()
        for i,agent in self.individual_information.items():
       
            current_state = [self.vector2angle(agent.states[-1])]
            neighbor_influence = []
            bias = []
            
            if agent.currently_listening:
                for neigh in agent.neighbors:       
                    neighboring_agent = self.individual_information[neigh]
                    if neighboring_agent.currently_speaking:
                        neighbor_influence.append(self.vector2angle(neighboring_agent.states[-1]))
            avg_angle = np.mean(current_state+neighbor_influence+bias)
            updated_states[i] = (cos(avg_angle),sin(avg_angle))
        return updated_states
    
    
    def vector2angle(self,v):
        return atan2(v[1],v[0])

    def update_reset_group(self,updated_states):
        for i,agent in self.individual_information.items():
            agent.states.append(updated_states[i])
            agent.spoke.append(agent.currently_speaking)
            agent.listened.append(agent.currently_listening)
            
            agent.currently_speaking = 0
            agent.currently_listening = 0
                
        
class Agent:
    def __init__(self,ID,initial_state):
        self.ID = ID
        self.states = [initial_state]
        
        self.currently_speaking = 0
        self.currently_listening = 0
        
        self.speaking_prob = []
        self.spoke = []
        
        self.listening_prob = []
        self.listened = []
        
        self.neighbors = None
    

def read_input_variable_file(filname):
    param_dict = dict()
    with open(filename, 'r') as f:
        for line in f:
            var,value = line.strip().split(':')
            if var.strip() == 'initial_state':
                if value.strip() == 'random':
                    param_dict['initial_state'] = lambda _: (np.random.random(),np.random.random())
                    continue
            param_dict[var.strip()] = float(value.strip())


    return param_dict()

def main(argv):
    print("hello world")
    params = read_input_variable_file(argv[1])
    network = nx.complete_graph(params['N'])

    group = Group_Conversations(network,params)
    group.initialize_groups()
    group.conversation_dynamics()

    all_times = {t:[] for t in range(801)}
    for i,agent in group.individual_information.items():
        for t in all_times:
            all_times[t].append(agent.states[t])
    t = []
    y = []
    for time,list1 in all_times.items():
        t.append(time)
        y.append(np.mean(list1))
    print(list(zip(t,y)))
    with open('attempt.txt','w') as f:
        for line in y:
            f.write('%1.4f\n' % line)
    return

if __name__ == "__main__":
    main(sys.argv)
    
    