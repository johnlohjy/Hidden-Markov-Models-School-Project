#!/usr/bin/env python
# coding: utf-8

# # Part 4 test: Design Challenge: Smoothing with Laplace Emission. Testing on ES and RU test.in

# In[1]:


import os
import copy


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


# #### Define function that estimates emission parameters with laplace smoothing with alpha on the unknown words

# In[3]:


def estimate_emission_parameters(training_set, k_value=1, alpha=1):
    state_count = {}
    state_observation_count = {}
    estimated_emission_parameters = {}
    #Get a set of the trained words
    trained_words = set()
    
    #k_value: k occurences of generating observation #UNK# from any label y
    
    for i in range(len(training_set)):
        #if its not a single empty line that separates sentences
        if(len(training_set[i])!=0):
            parts = training_set[i].split(" ")
            observation = ' '.join(parts[:len(parts)-1])
            state = parts[-1]
            
            #Increment count(y->x)
            if (observation,state) in state_observation_count:
                state_observation_count[(observation,state)]+=1
            else:
                state_observation_count[(observation,state)]=1

            #Increment count(y)
            if state in state_count:
                state_count[state]+=1
            else:
                state_count[state]=1
                
            trained_words.add(observation)
        else:
            continue
        
    vocabulary_size = len(trained_words)
    
    #Use Laplace smoothing with alpha
    for k,v in state_observation_count.items():
        estimated_emission_parameters[k] = v/(state_count[k[1]]+k_value)
        estimated_emission_parameters[("#UNK#",k[1])] = (k_value+alpha)/( (state_count[k[1]]+k_value) + alpha*vocabulary_size)
    
    return estimated_emission_parameters, list(trained_words) 


# #### Define function that estimates transition parameters

# In[4]:


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

# #### Define function that implements the viterbi algorithm

# In[5]:


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

# In[6]:


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

# In[7]:


filepath_ES_train = os.path.join(os.getcwd(), 'Data', 'ES', 'train')

#Read the file contents
with open(filepath_ES_train, 'r', encoding='utf-8') as file:
    file_contents_ES_train = file.readlines()
    
#Convert to training set
es_training_set = [w.strip() for w in file_contents_ES_train]


# #### Learn ES emission and transition parameters

# In[8]:


#Calculate the parameters using the training set
estimated_emission_parameters,trained_words = estimate_emission_parameters(es_training_set)
estimated_transition_parameters, all_states = estimate_transition_parameters(es_training_set)


# #### Read test.in Dataset for evaluation

# In[9]:


filepath_ES_testin = os.path.join(os.getcwd(), 'Data', 'ES', 'test.in')

#Read the file contents
with open(filepath_ES_testin, 'r', encoding='utf-8') as file:
    file_contents_ES_testin = file.readlines()
    
es_testin = [w.strip() for w in file_contents_ES_testin]


# #### Convert ES test.in into a list of lists where each list a sentence of observations

# In[10]:


es_testin = sentence_creator_observations(es_testin)


# #### Run the Vertibi Algorithm on each sentence of ES test.in

# In[11]:


#For each sentence
for i in range(len(es_testin)):
    es_testin[i] = vertibi_algo(es_testin[i], all_states, trained_words, estimated_emission_parameters, estimated_transition_parameters)


# #### Join all the results

# In[12]:


es_testin_predicted = []
for sentence in es_testin:
    for i in range(len(sentence)):
        if(i==len(sentence)-1):
            es_testin_predicted.append(sentence[i])
            es_testin_predicted.append('')
        else:
            es_testin_predicted.append(sentence[i])


# #### Write to test.p4.out

# In[13]:


filepath_test_p4_out = os.path.join(os.getcwd(), 'Data', 'ES', 'test.p4.out')


# In[14]:


with open(filepath_test_p4_out, 'w', encoding='utf-8') as file:
    for line in es_testin_predicted:
        file.write(line + '\n')


# ## Train and Evaluate with RU

# #### Read RU Train Dataset

# In[15]:


filepath_RU_train = os.path.join(os.getcwd(), 'Data', 'RU', 'train')

#Read the file contents
with open(filepath_RU_train, 'r', encoding='utf-8') as file:
    file_contents_RU_train = file.readlines()
    
#Convert to training set
ru_training_set = [w.strip() for w in file_contents_RU_train]


# #### Learn RU emission and transition parameters

# In[16]:


#Calculate the parameters using the training set
estimated_emission_parameters,trained_words = estimate_emission_parameters(ru_training_set)
estimated_transition_parameters, all_states = estimate_transition_parameters(ru_training_set)


# #### Read RU test.in Dataset for evaluation

# In[17]:


filepath_RU_testin = os.path.join(os.getcwd(), 'Data', 'RU', 'test.in')

#Read the file contents
with open(filepath_RU_testin, 'r', encoding='utf-8') as file:
    file_contents_RU_testin = file.readlines()
    
ru_testin = [w.strip() for w in file_contents_RU_testin]


# #### Convert RU test.in into a list of lists where each list a sentence of observations

# In[18]:


ru_testin = sentence_creator_observations(ru_testin)


# #### Run the Vertibi Algorithm on each sentence of RU test.in

# In[19]:


#For each sentence
for i in range(len(ru_testin)):
    ru_testin[i] = vertibi_algo(ru_testin[i], all_states, trained_words, estimated_emission_parameters, estimated_transition_parameters)


# #### Join all the results

# In[20]:


ru_testin_predicted = []
for sentence in ru_testin:
    for i in range(len(sentence)):
        if(i==len(sentence)-1):
            ru_testin_predicted.append(sentence[i])
            ru_testin_predicted.append('')
        else:
            ru_testin_predicted.append(sentence[i])


# #### Write to test.p4.out

# In[21]:


filepath_test_p4_out = os.path.join(os.getcwd(), 'Data', 'RU', 'test.p4.out')


# In[22]:


with open(filepath_test_p4_out, 'w', encoding='utf-8') as file:
    for line in ru_testin_predicted:
        file.write(line + '\n')

