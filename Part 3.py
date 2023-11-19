#!/usr/bin/env python
# coding: utf-8

# # Part 3: Implement an algorithm to find the k-th best output sequences (25 Points)

# In[1]:


import os
import copy
import heapq
from project_utils import estimate_emission_parameters
from project_utils import sentence_creator_states
from project_utils import estimate_transition_parameters
from project_utils import sentence_creator_observations


# ### Define the modified viterbi function to compute k best sequences using the forward pass with beam search, Comparison of sequences is done by joint probability, the products of transition and emissions

# #### Function that keeps only the k best sequences after every position is done

# In[2]:


def modified_vertibi_algo(sentence_original, all_states_original, trained_words_original, emission_parameters_original, transition_parameters_original, k_sequences):
    #Create deep copies
    sentence = copy.deepcopy(sentence_original)
    all_states = copy.deepcopy(all_states_original)
    trained_words = copy.deepcopy(trained_words_original)
    emission_parameters = copy.deepcopy(emission_parameters_original)
    transition_parameters = copy.deepcopy(transition_parameters_original)
    all_states.append('STOP')
    
    #1. Initialisation Step: Initialise a heap to store the sequences containing 
    #   only the score of the START state and the START state itself
    top_sequences = [(1, ['START'])]
    
    # Initialise a variable to keep track of the position in seq_path
    seq_path_position = 0
    
    #2. Forward Pass from the first word to nth word (inclusive)
    # Recursive step of beam search 
    # For each position j from the first word to nth word (inclusive)
    for position in range(0, len(sentence)):

        # Initialise a list to hold all temp sequences
        temp_sequences = []

        # For each sequence score and sequence path in the current top sequences
        for seq_score, seq_path in top_sequences:

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
                        heapq.heappush(temp_sequences, extended_seq)    


        #Prune step: Get the k top_sequences from this pass only
        top_sequences = heapq.nlargest(k_sequences, temp_sequences)
        #Increase the seq_path_position counter
        seq_path_position+=1
        
    #3. Recursion step of beam search at the last word
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
    
    #Check if top_sequences is empty. 
    #If it is, expand the beam search to include all possible sequences
    #As some sequences that were potential optimal candidates may have gotten dropped earlier on
    #as they did not have a high score in the earlier positions
    if(len(top_sequences)==0):
        top_sequences = modified_vertibi_algo_allseq(sentence_original, all_states_original, trained_words_original, emission_parameters_original, transition_parameters_original, k_sequences)
        return top_sequences
    
    #In the case where the number of sequences generated is less than the number of sequences
    #Pad the sequences using the last i.e. worst performing sequence
    while(len(top_sequences)<k_sequences):
        top_sequences.append(copy.deepcopy(top_sequences[len(top_sequences)-1]))
    
    # Clean up the top_sequences to get only the predicted states
    # for the top k sequences
    for i in range(len(top_sequences)):
        temp = top_sequences[i][1]
        temp.pop(0)
        temp.pop(len(temp)-1)
        top_sequences[i] = temp
        
    # Combine the sentences with its predicted states
    for i in range(len(top_sequences)):
        temp = [f"{sentence[j]} {top_sequences[i][j]}" for j in range(len(sentence))]
        top_sequences[i] = temp
        
    return top_sequences


# #### Function that keeps all sequences after every position is done

# In[3]:


def modified_vertibi_algo_allseq(sentence_original, all_states_original, trained_words_original, emission_parameters_original, transition_parameters_original, k_sequences):
    #Create deep copies
    sentence = copy.deepcopy(sentence_original)
    all_states = copy.deepcopy(all_states_original)
    trained_words = copy.deepcopy(trained_words_original)
    emission_parameters = copy.deepcopy(emission_parameters_original)
    transition_parameters = copy.deepcopy(transition_parameters_original)
    all_states.append('STOP')
    
    #1. Initialisation Step: Initialise a heap to store the sequences containing 
    #   only the score of the START state and the START state itself
    all_sequences = [(1, ['START'])]
    
    # Initialise a variable to keep track of the position in seq_path
    seq_path_position = 0
    
    #2. Forward Pass from the first word to nth word (inclusive)
    # Recursive step of beam search 
    # For each position j from the first word to nth word (inclusive)
    for position in range(0, len(sentence)):

        # Initialise a list to hold all temp sequences
        temp_sequences = []

        # For each sequence score and sequence path in the current top sequences
        for seq_score, seq_path in all_sequences:

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
                        heapq.heappush(temp_sequences, extended_seq)    


        all_sequences = temp_sequences
        #Increase the seq_path_position counter
        seq_path_position+=1
        
    #3. Recursion step of beam search at the last word
    # Initialise a list to hold all temp sequences
    temp_sequences = []

    # For each sequence score and sequence path in the current sequences found
    for seq_score, seq_path in all_sequences:

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


    #Prune step: Get the final top 8 sequences from all sequences 
    top_sequences = heapq.nlargest(k_sequences, temp_sequences)
    
    #If no prediction is available for this sentence, assign a None value for all the states
    if(len(top_sequences)==0):
        top_sequences = [(0, ['START'] + [None]*len(sentence) + ['STOP'])]
    
    #In the case where the number of sequences generated is less than the number of sequences
    #Pad the sequences using the last i.e. worst performing sequence
    while(len(top_sequences)<k_sequences):
        top_sequences.append(copy.deepcopy(top_sequences[len(top_sequences)-1]))
    
    # Clean up the top_sequences to get only the predicted states
    # for the top k sequences
    for i in range(len(top_sequences)):
        temp = top_sequences[i][1]
        temp.pop(0)
        temp.pop(len(temp)-1)
        top_sequences[i] = temp
        
    # Combine the sentences with its predicted states
    for i in range(len(top_sequences)):
        temp = [f"{sentence[j]} {top_sequences[i][j]}" for j in range(len(sentence))]
        top_sequences[i] = temp
        
    return top_sequences


# ## Train and Evaluate with ES

# #### Read ES Train Dataset

# In[4]:


filepath_ES_train = os.path.join(os.getcwd(), 'Data', 'ES', 'train')

#Read the file contents
with open(filepath_ES_train, 'r', encoding='utf-8') as file:
    file_contents_ES_train = file.readlines()
    
#Convert to training set
es_training_set = [w.strip() for w in file_contents_ES_train]


# #### Learn ES emission and transition parameters

# In[5]:


#Calculate the parameters using the training set
estimated_emission_parameters,trained_words = estimate_emission_parameters(es_training_set)
estimated_transition_parameters, all_states = estimate_transition_parameters(es_training_set)


# #### Read ES dev.in Dataset for evaluation

# In[6]:


filepath_ES_devin = os.path.join(os.getcwd(), 'Data', 'ES', 'dev.in')

#Read the file contents
with open(filepath_ES_devin, 'r', encoding='utf-8') as file:
    file_contents_ES_devin = file.readlines()
    
es_devin = [w.strip() for w in file_contents_ES_devin]


# #### Convert ES dev.in into a list of lists where each list a sentence of observations

# In[7]:


es_devin = sentence_creator_observations(es_devin)


# #### Run the Modified Vertibi Algorithm on each sentence of ES dev.in

# In[8]:


k_sequences = 8


# In[9]:


#For each sentence
for i in range(len(es_devin)):
    es_devin[i] = modified_vertibi_algo(es_devin[i], all_states, trained_words, estimated_emission_parameters, estimated_transition_parameters, k_sequences)


# #### Join all k sequences together

# In[10]:


es_devin_predicted_k_sequences = []
# For the k-th best sequence
for i in range(k_sequences):
    # Define a list to hold the results for the k-th best sequence
    k_sequence = []
    
    # For k-th best sequence of each sentence
    for sequences in es_devin:
        # For each pair of word and predicted tag in the k-th best sequence of the current sentence
        # Add it to k_sequence
        for j in range(len(sequences[i])):
            # If its the last pair of word and predicted tag, add an empty line behind
            if(j==len(sequences[i])-1):
                k_sequence.append(sequences[i][j])
                k_sequence.append("")
            # Add the word and predicted tag
            else:
                k_sequence.append(sequences[i][j])
    
    # When this k-th best sequence is done, move to the next k-th best sequence
    es_devin_predicted_k_sequences.append(k_sequence)


# #### Results for ES

# #### Write to dev.p3.2nd.out

# In[11]:


filepath_dev_p3_k2_out = os.path.join(os.getcwd(), 'Data', 'ES', 'dev.p3.2nd.out')
with open(filepath_dev_p3_k2_out, 'w', encoding='utf-8') as file:
    for line in es_devin_predicted_k_sequences[1]:
        file.write(line + '\n')

python evalResult.py dev.out dev.p3.2nd.out#Entity in gold data: 229
#Entity in prediction: 454

#Correct Entity : 119
Entity  precision: 0.2621
Entity  recall: 0.5197
Entity  F: 0.3485

#Correct Sentiment : 70
Sentiment  precision: 0.1542
Sentiment  recall: 0.3057
Sentiment  F: 0.2050
# #### Write to dev.p3.8th.out

# In[12]:


filepath_dev_p3_k8_out = os.path.join(os.getcwd(), 'Data', 'ES', 'dev.p3.8th.out')
with open(filepath_dev_p3_k8_out, 'w', encoding='utf-8') as file:
    for line in es_devin_predicted_k_sequences[7]:
        file.write(line + '\n')

python evalResult.py dev.out dev.p3.8th.out#Entity in gold data: 229
#Entity in prediction: 539

#Correct Entity : 106
Entity  precision: 0.1967
Entity  recall: 0.4629
Entity  F: 0.2760

#Correct Sentiment : 63
Sentiment  precision: 0.1169
Sentiment  recall: 0.2751
Sentiment  F: 0.1641
# ## Train and Evaluate with RU

# #### Read RU Train Dataset

# In[13]:


filepath_RU_train = os.path.join(os.getcwd(), 'Data', 'RU', 'train')

#Read the file contents
with open(filepath_RU_train, 'r', encoding='utf-8') as file:
    file_contents_RU_train = file.readlines()
    
#Convert to training set
ru_training_set = [w.strip() for w in file_contents_RU_train]


# #### Learn RU emission and transition parameters

# In[14]:


#Calculate the parameters using the training set
estimated_emission_parameters,trained_words = estimate_emission_parameters(ru_training_set)
estimated_transition_parameters, all_states = estimate_transition_parameters(ru_training_set)


# #### Read RU dev.in Dataset for evaluation

# In[15]:


filepath_RU_devin = os.path.join(os.getcwd(), 'Data', 'RU', 'dev.in')

#Read the file contents
with open(filepath_RU_devin, 'r', encoding='utf-8') as file:
    file_contents_RU_devin = file.readlines()
    
ru_devin = [w.strip() for w in file_contents_RU_devin]


# #### Convert RU dev.in into a list of lists where each list a sentence of observations

# In[16]:


ru_devin = sentence_creator_observations(ru_devin)


# #### Run the Modified Vertibi Algorithm on each sentence of ES dev.in

# In[17]:


k_sequences = 8


# In[18]:


#For each sentence
for i in range(len(ru_devin)):
    ru_devin[i] = modified_vertibi_algo(ru_devin[i], all_states, trained_words, estimated_emission_parameters, estimated_transition_parameters, k_sequences)


# #### Join all k sequences together

# In[19]:


ru_devin_predicted_k_sequences = []
# For the k-th best sequence
for i in range(k_sequences):
    # Define a list to hold the results for the k-th best sequence
    k_sequence = []
    
    # For k-th best sequence of each sentence
    for sequences in ru_devin:
        # For each pair of word and predicted tag in the k-th best sequence of the current sentence
        # Add it to k_sequence
        for j in range(len(sequences[i])):
            # If its the last pair of word and predicted tag, add an empty line behind
            if(j==len(sequences[i])-1):
                k_sequence.append(sequences[i][j])
                k_sequence.append("")
            # Add the word and predicted tag
            else:
                k_sequence.append(sequences[i][j])
    
    # When this k-th best sequence is done, move to the next k-th best sequence
    ru_devin_predicted_k_sequences.append(k_sequence)


# #### Results for RU

# #### Write to dev.p3.2nd.out

# In[20]:


filepath_dev_p3_k2_out = os.path.join(os.getcwd(), 'Data', 'RU', 'dev.p3.2nd.out')
with open(filepath_dev_p3_k2_out, 'w', encoding='utf-8') as file:
    for line in ru_devin_predicted_k_sequences[1]:
        file.write(line + '\n')

python evalResult.py dev.out dev.p3.2nd.out#Entity in gold data: 389
#Entity in prediction: 677

#Correct Entity : 198
Entity  precision: 0.2925
Entity  recall: 0.5090
Entity  F: 0.3715

#Correct Sentiment : 123
Sentiment  precision: 0.1817
Sentiment  recall: 0.3162
Sentiment  F: 0.2308
# #### Write to dev.p3.8th.out

# In[21]:


filepath_dev_p3_k8_out = os.path.join(os.getcwd(), 'Data', 'RU', 'dev.p3.8th.out')
with open(filepath_dev_p3_k8_out, 'w', encoding='utf-8') as file:
    for line in ru_devin_predicted_k_sequences[7]:
        file.write(line + '\n')

python evalResult.py dev.out dev.p3.8th.out#Entity in gold data: 389
#Entity in prediction: 779

#Correct Entity : 176
Entity  precision: 0.2259
Entity  recall: 0.4524
Entity  F: 0.3014

#Correct Sentiment : 101
Sentiment  precision: 0.1297
Sentiment  recall: 0.2596
Sentiment  F: 0.1729
# In[ ]:




