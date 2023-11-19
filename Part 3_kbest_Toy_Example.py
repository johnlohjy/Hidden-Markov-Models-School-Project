#!/usr/bin/env python
# coding: utf-8

# # Part 3 Toy Example for the modified viterbi function to compute k best sequences using the forward pass with beam search, Comparison of sequences is done by joint probability, the products of transition and emissions

# ### For Function that keeps only the k best sequences after every position is done

# In[1]:


import os
import copy
import heapq
from project_utils import estimate_emission_parameters
from project_utils import sentence_creator_states
from project_utils import estimate_transition_parameters
from project_utils import sentence_creator_observations


# ### Learn emission and transition parameters from training set

# #### Read training set

# In[2]:


filepath_ES_train = os.path.join(os.getcwd(), 'Data', 'ES', 'train')

#Read the file contents
with open(filepath_ES_train, 'r', encoding='utf-8') as file:
    file_contents_ES_train = file.readlines()
    
#Convert to training set
es_training_set = [w.strip() for w in file_contents_ES_train]


# In[3]:


#Calculate the parameters using the training set
estimated_emission_parameters,trained_words = estimate_emission_parameters(es_training_set)


# In[4]:


estimated_transition_parameters, all_states = estimate_transition_parameters(es_training_set)


# ### Implement Modified Viterbi Algorithm to find k-best sequences

# In[5]:


filepath_ES_devin = os.path.join(os.getcwd(), 'Data', 'ES', 'dev.in')

#Read the file contents
with open(filepath_ES_devin, 'r', encoding='utf-8') as file:
    file_contents_ES_devin = file.readlines()
    
es_devin = [w.strip() for w in file_contents_ES_devin]


# #### Convert ES dev.in to a list of lists where each list is a sentence of observations

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


# In[7]:


es_devin = sentence_creator_observations(es_devin)


# ### Test Modified Vertibi Algorithm with Toy Example

# #### Get the first sentence

# In[8]:


sentence = es_devin[0]


# #### Append STOP state

# In[9]:


all_states.append('STOP')


# #### Deep copy learnt parameters from ES train earlier

# In[10]:


emission_parameters = copy.deepcopy(estimated_emission_parameters)


# In[11]:


transition_parameters = copy.deepcopy(estimated_transition_parameters)


# #### Initialisation Step

# In[12]:


k_sequences = 8


# In[13]:


#1. Initialisation Step: Initialise a heap to store the sequences containing 
#   only the score of the START state and the START state itself
top_sequences = [(1, ['START'])]


# #### Recursive Step of beam search from the first word to nth word (inclusive)

# In[14]:


# Initialise a variable to keep track of the position in seq_path
seq_path_position = 0

#2. Forward Pass from the first word to nth word (inclusive)
# Recursive step of beam search 
# For each position j from the first word to nth word (inclusive)
# For each position j from the first word to nth word (inclusive)
for position in range(0, len(sentence)):
    
    print(f"LOOKING AT POSITION {position}")
        
    # Initialise a list to hold all temp sequences
    temp_sequences = []
    
    # For each sequence score and sequence path in the current top sequences
    for seq_score, seq_path in top_sequences:
        print(f"Sequence Path: {seq_path} with Sequence Score: {seq_score}")
        print(" ")

        # For each state u belonging to T at the current position except for START and STOP
        for state in all_states:
            if(state=='START' or state=='STOP'):
                continue

            # If the word appears in the training set
            if(sentence[position] in trained_words):
                # If the emission and transition has been trained before
                if((sentence[position],state) in emission_parameters.keys() and (seq_path[seq_path_position],state) in transition_parameters.keys()):
                    # Calculate the extended sequence score: 
                    # score of current seq (product of transitions and emissions in seq) *
                    # transiton prob of latest state in seq to curr_state *
                    # emission prob of observation from curr_state 
                    extended_seq_score = seq_score*transition_parameters[(seq_path[seq_path_position],state)]*emission_parameters[(sentence[position],state)]
                    
                    # Extend the sequence path
                    extended_seq_path = seq_path + [state]
                    
                    # Push the extended sequence into the heap
                    extended_seq = (extended_seq_score,extended_seq_path)
                    print(f"At state: {state}")
                    print(f"Managed to get Extended Sequence Path: {extended_seq_path} with Extended Sequence Score: {extended_seq_score}")
                    print("Pushing extended sequence into heap")
                    print(" ")
                    heapq.heappush(temp_sequences, extended_seq)
            else:
                # If the emission and transition has been trained before
                if(("#UNK#",state) in emission_parameters.keys() and (seq_path[seq_path_position],state) in transition_parameters.keys()):
                    # Calculate the extended sequence score: 
                    # score of current seq (product of transitions and emissions in seq) *
                    # transiton prob of latest state in seq to curr_state *
                    # emission prob of #UNK# from curr_state
                    extended_seq_score = seq_score*transition_parameters[(seq_path[seq_path_position],state)]*emission_parameters[("#UNK#",state)]
                    
                    # Extend the sequence path
                    extended_seq_path = seq_path + [state]
                    
                    # Push the extended sequence into the heap
                    extended_seq = (extended_seq_score,extended_seq_path)
                    print(f"At state: {state}")
                    print(f"Managed to get Extended Sequence Path: {extended_seq_path} with Extended Sequence Score: {extended_seq_score}")
                    print("Pushing extended sequence into heap")
                    print(" ")
                    heapq.heappush(temp_sequences, extended_seq)
        print("##############################################")     
                
    #Prune step: Get the k top_sequences from this pass only
    top_sequences = heapq.nlargest(k_sequences, temp_sequences)
    
    print(f"Top Sequences at Position: {position}:")
    print(top_sequences)
    print("############################################")
    print("############################################")
    print(" ")
    print(" ")
    #Increase the seq_path_position counter
    seq_path_position+=1


# In[15]:


top_sequences


# #### Recursive Step of beam search at the last word

# In[16]:


# Initialise a list to hold all temp sequences
temp_sequences = []

# For each sequence score and sequence path in the current top sequences
for seq_score, seq_path in top_sequences:

    # If the transition has been trained before
    if((seq_path[seq_path_position],"STOP") in transition_parameters.keys()):
        # Calculate the extended sequence score: 
        # score of current seq (product of transitions and emissions in seq) *
        # transiton prob of latest state in seq to STOP state  
        extended_seq_score = seq_score*transition_parameters[(seq_path[seq_path_position],state)]

        # Extend the sequence path
        extended_seq_path = seq_path + ["STOP"]

        # Push the extended sequence into the heap
        extended_seq = (extended_seq_score,extended_seq_path)
        heapq.heappush(temp_sequences, extended_seq)


#Prune step: Get the final top 8 sequences 
top_sequences = heapq.nlargest(k_sequences, temp_sequences)


# In[17]:


top_sequences


# In[18]:


#In the case where the number of sequences generated is less than the number of sequences
#Pad the sequences using the last i.e. worst performing sequence
while(len(top_sequences)<8):
    top_sequences.append(copy.deepcopy(top_sequences[len(top_sequences)-1]))


# In[19]:


top_sequences


# In[20]:


# Clean up the top_sequences to get only the predicted states
# for the top k sequences
for i in range(len(top_sequences)):
    temp = top_sequences[i][1]
    temp.pop(0)
    temp.pop(len(temp)-1)
    top_sequences[i] = temp


# In[21]:


top_sequences


# In[22]:


# Combine the sentences with its predicted states
for i in range(len(top_sequences)):
    temp = [f"{sentence[j]} {top_sequences[i][j]}" for j in range(len(sentence))]
    top_sequences[i] = temp


# In[23]:


top_sequences


# In[ ]:




