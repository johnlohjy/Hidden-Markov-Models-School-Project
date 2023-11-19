#!/usr/bin/env python
# coding: utf-8

# # Part 2.1: Write a function that estimates the transition parameters from the training set using MLE (10 points)

# In[1]:


import os
import copy
from project_utils import estimate_emission_parameters


# #### Define helper function that creates a list of lists where each list contains only the states of a sentence

# In[2]:


#Function to create a list of lists where each list contains only the states of a sentence
def sentence_creator_states(original):
    sentences = []
    sentence = []
    for i in original:
        #If it is not an empty line
        if i!='':
            #Append the state to sentence
            parts = i.split(" ")
            state = parts[-1]
            sentence.append(state)
        #If it is an empty line
        #The sentence is complete
        else:
            #Add the START and STOP state for computation later
            sentence.insert(0, "START")
            sentence.append("STOP")
            sentences.append(sentence)
            sentence = []
            
    #In the case of the last sentence where it the training set does not end with an empty line
    if len(sentence)!=0:
        sentences.append(sentence)
    
    return sentences


# #### Define function that estimates transition parameters

# In[3]:


def estimate_transition_parameters(training_set):
    #Break the training set into sentences, where each sentence contains its states
    training_set = sentence_creator_states(training_set)
    state_count = {}
    state_transition_count = {}
    estimated_transition_parameters = {}
    
    #For each sentence (list of states)
    for sentence in training_set:
        #For each state in the sentence
        for i in range(len(sentence)):
            #Compute the counts up until the STOP state
            if(i!=len(sentence)-1):
                #Increment count(y_i-1 -> y_i)
                if (sentence[i], sentence[i+1]) in state_transition_count:
                    state_transition_count[(sentence[i], sentence[i+1])]+=1
                else:
                    state_transition_count[(sentence[i], sentence[i+1])]=1

                #Increment count(y_i-1)
                if sentence[i] in state_count:
                    state_count[sentence[i]]+=1
                else:
                    state_count[sentence[i]]=1
                    
    #For each y_i|y_i-1, calculate count(y_i-1 -> y_i)/count(y_i-1)
    for k,v in state_transition_count.items():
        estimated_transition_parameters[k] = v/state_count[k[0]]
    
    return estimated_transition_parameters, list(state_count.keys())


# # Part 2.2: Implement the Viterbi Algorithm

# #### Define function to perform viterbi algorithm

# In[4]:


def vertibi_algo(sentence_original, all_states_original, trained_words_original, emission_parameters_original, transition_parameters_original):
    #Create deep copies
    sentence = copy.deepcopy(sentence_original)
    all_states = copy.deepcopy(all_states_original)
    trained_words = copy.deepcopy(trained_words_original)
    emission_parameters = copy.deepcopy(emission_parameters_original)
    transition_parameters = copy.deepcopy(transition_parameters_original)
    all_states.append('STOP')
    
    #Pad each sentence to account for position 0 (start) and position n+1 (stop)
    sentence.insert(0, "padding")
    sentence.append("padding")
    
    #Initialise a score matrix to store the highest scoring paths 
    #from START to state u at position j
    #index by [position,state]
    #number of states T
    #number of positions 0 to n+1
    #dimensions: number of positions x number of states
    score_matrix = [[0 for _ in range(len(all_states))] for _ in range(len(sentence))] 
    
    #FORWARD PASS
    #0. Map each state to an index number
    all_states_map = {k:v for v,k in enumerate(all_states)}
    
    #1. Initialisation Step
    for state in all_states:
        if(state=='START'):
            score_matrix[0][all_states_map[state]] = 1
        else:
            score_matrix[0][all_states_map[state]] = 0
            
    #2. Forward Pass from j=0 to j=n-1 (inclusive)
    #0 1 n-1 n n+1
    #len = 5
    #len-2 = 3
    #For each position j from 0 to n-1
    for position in range(len(sentence)-2):
        #For each state u belonging to T except for START and STOP
        for state in all_states:
            if(state=='START' or state=='STOP'):
                continue
            score = 0
            #For each prev state v belonging to T
            for prev_state in all_states:
                temp = 0
                #If the word in j+1 position appears in the training set
                if(sentence[position+1] in trained_words):
                    #If the emission and transition has been trained before
                    if((sentence[position+1],state) in emission_parameters.keys() and (prev_state,state) in transition_parameters.keys()):
                        #Calculate: score of prev_state v in position j * emission prob of observation from curr_state u * transiton prob of prev_state v to curr_state u
                        temp = score_matrix[position][all_states_map[prev_state]]*emission_parameters[(sentence[position+1],state)]*transition_parameters[(prev_state,state)]
                else:
                    #If the emission and transition has been trained before
                    if(("#UNK#",state) in emission_parameters.keys() and (prev_state,state) in transition_parameters.keys()):
                        #Calculate: score of prev_state v in position j * emission prob of #UNK# from curr_state u * transiton prob of prev_state v to curr_state u
                        temp = score_matrix[position][all_states_map[prev_state]]*emission_parameters[("#UNK#",state)]*transition_parameters[(prev_state,state)]
                #Store the score of the highest scoring path (max v) from START to this node (position j+1, state u) 
                if(temp>score):
                    score = temp
                    score_matrix[position+1][all_states_map[state]] = score
                    
    #3. Final Step
    #0 1 n-1 n n+1
    #len = 5
    #len-2 = 3
    #len-1 = 4
    score = 0
    #For each prev state v belonging to T
    for prev_state in all_states:
        temp = 0
        #If the transition has been trained before
        if((prev_state,'STOP') in transition_parameters.keys()):
            #Calculate: score of prev_state v in position n * transiton prob of prev_state v to STOP state
            temp = score_matrix[len(sentence)-2][all_states_map[prev_state]]*transition_parameters[(prev_state,'STOP')]

        #Store the score of the highest scoring path (max v) from START to STOP 
        if(temp>score):
            score = temp
            score_matrix[len(sentence)-1][all_states_map['STOP']] = score
            
    #############################################################################################
    
    #Backward Pass
    #Define optimal_y: a list to store the optimal y found in the backwards pass
    optimal_y = [None for _ in range(len(sentence))]
    
    #len(sentence)-1: last index
    #Calculate y_n*
    #at position n
    score = 0
    for state in all_states:
        temp = 0
        if(state=='START' or state=='STOP'):
            continue
        #If the transition has been trained before
        if((state,'STOP') in transition_parameters.keys()):
            temp = score_matrix[len(sentence)-2][all_states_map[state]] * transition_parameters[(state,'STOP')]
        if(temp>score):
            score = temp
            optimal_y[len(sentence)-2] = state

    #Calculate y_j*
    #from position n-1 to position 1 (inclusive)
    for j in range(len(sentence)-3,0,-1):
        score = 0
        for state in all_states:
            temp = 0
            if(state=='START' or state=='STOP'):
                continue
            #If the transition has been trained before: from this state to the optimal state found in the next position
            if((state,optimal_y[j+1]) in transition_parameters.keys()):
                temp = score_matrix[j][all_states_map[state]] * transition_parameters[(state,optimal_y[j+1])] 
            if(temp>score):
                score = temp
                optimal_y[j] = state
                
    #############################################################################################
    
    #Clean up
    
    #Remove the added padding to the sentence
    sentence.pop(0)
    sentence.pop(len(sentence)-1)
    
    #Remove position 0 and position n+1
    optimal_y.pop(0)
    optimal_y.pop(len(optimal_y)-1)
    
    #Join the sentence with its predicted states
    tagged_sentence = [f"{sentence[i]} {optimal_y[i]}" for i in range(len(sentence))]
    
    return tagged_sentence


# #### Define helper function that creates a list of lists where each list contains only the observations of a sentence

# In[5]:


#Function to create a list of lists where each list contains only the observations of a sentence
def sentence_creator_observations(original):
    sentences = []
    sentence = []
    for i in original:
        if i!='':
            sentence.append(i)
        else:
            sentences.append(sentence)
            sentence = []
            
    if len(sentence)!=0:
        sentences.append(sentence)
    
    return sentences


# ## Train and Evaluate with ES

# #### Read ES Train Dataset

# In[6]:


filepath_ES_train = os.path.join(os.getcwd(), 'Data', 'ES', 'train')

#Read the file contents
with open(filepath_ES_train, 'r', encoding='utf-8') as file:
    file_contents_ES_train = file.readlines()
    
#Convert to training set
es_training_set = [w.strip() for w in file_contents_ES_train]


# #### Learn ES emission and transition parameters

# In[7]:


#Calculate the parameters using the training set
estimated_emission_parameters,trained_words = estimate_emission_parameters(es_training_set)
estimated_transition_parameters, all_states = estimate_transition_parameters(es_training_set)


# #### Read ES dev.in Dataset for evaluation

# In[8]:


filepath_ES_devin = os.path.join(os.getcwd(), 'Data', 'ES', 'dev.in')

#Read the file contents
with open(filepath_ES_devin, 'r', encoding='utf-8') as file:
    file_contents_ES_devin = file.readlines()
    
es_devin = [w.strip() for w in file_contents_ES_devin]


# #### Convert ES dev.in into a list of lists where each list a sentence of observations

# In[9]:


es_devin = sentence_creator_observations(es_devin)


# #### Run the Vertibi Algorithm on each sentence of ES dev.in

# In[10]:


#For each sentence
for i in range(len(es_devin)):
    es_devin[i] = vertibi_algo(es_devin[i], all_states, trained_words, estimated_emission_parameters, estimated_transition_parameters)


# #### Join all the results

# In[11]:


es_devin_predicted = []
for sentence in es_devin:
    for i in range(len(sentence)):
        if(i==len(sentence)-1):
            es_devin_predicted.append(sentence[i])
            es_devin_predicted.append('')
        else:
            es_devin_predicted.append(sentence[i])


# #### Write to dev.p2.out

# In[12]:


filepath_dev_p2_out = os.path.join(os.getcwd(), 'Data', 'ES', 'dev.p2.out')


# In[13]:


with open(filepath_dev_p2_out, 'w', encoding='utf-8') as file:
    for line in es_devin_predicted:
        file.write(line + '\n')


# #### Compare dev.p2.out with dev.out for ES
python evalResult.py dev.out dev.p2.out#Entity in gold data: 229
#Entity in prediction: 542

#Correct Entity : 134
Entity  precision: 0.2472
Entity  recall: 0.5852
Entity  F: 0.3476

#Correct Sentiment : 97
Sentiment  precision: 0.1790
Sentiment  recall: 0.4236
Sentiment  F: 0.2516
# ## Train and Evaluate with RU

# #### Read RU Train Dataset

# In[14]:


filepath_RU_train = os.path.join(os.getcwd(), 'Data', 'RU', 'train')

#Read the file contents
with open(filepath_RU_train, 'r', encoding='utf-8') as file:
    file_contents_RU_train = file.readlines()
    
#Convert to training set
ru_training_set = [w.strip() for w in file_contents_RU_train]


# #### Learn RU emission and transition parameters

# In[15]:


#Calculate the parameters using the training set
estimated_emission_parameters,trained_words = estimate_emission_parameters(ru_training_set)
estimated_transition_parameters, all_states = estimate_transition_parameters(ru_training_set)


# #### Read RU dev.in Dataset for evaluation

# In[16]:


filepath_RU_devin = os.path.join(os.getcwd(), 'Data', 'RU', 'dev.in')

#Read the file contents
with open(filepath_RU_devin, 'r', encoding='utf-8') as file:
    file_contents_RU_devin = file.readlines()
    
ru_devin = [w.strip() for w in file_contents_RU_devin]


# #### Convert RU dev.in into a list of lists where each list a sentence of observations

# In[17]:


ru_devin = sentence_creator_observations(ru_devin)


# #### Run the Vertibi Algorithm on each sentence of RU dev.in

# In[18]:


#For each sentence
for i in range(len(ru_devin)):
    ru_devin[i] = vertibi_algo(ru_devin[i], all_states, trained_words, estimated_emission_parameters, estimated_transition_parameters)


# #### Join all the results

# In[19]:


ru_devin_predicted = []
for sentence in ru_devin:
    for i in range(len(sentence)):
        if(i==len(sentence)-1):
            ru_devin_predicted.append(sentence[i])
            ru_devin_predicted.append('')
        else:
            ru_devin_predicted.append(sentence[i])


# #### Write to dev.p2.out

# In[20]:


filepath_dev_p2_out = os.path.join(os.getcwd(), 'Data', 'RU', 'dev.p2.out')


# In[21]:


with open(filepath_dev_p2_out, 'w', encoding='utf-8') as file:
    for line in ru_devin_predicted:
        file.write(line + '\n')


# #### Compare dev.p2.out with dev.out for RU
python evalResult.py dev.out dev.p2.out#Entity in gold data: 389
#Entity in prediction: 484

#Correct Entity : 188
Entity  precision: 0.3884
Entity  recall: 0.4833
Entity  F: 0.4307

#Correct Sentiment : 129
Sentiment  precision: 0.2665
Sentiment  recall: 0.3316
Sentiment  F: 0.2955