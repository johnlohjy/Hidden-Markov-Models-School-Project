#!/usr/bin/env python
# coding: utf-8

# # Part 1.1: Write a function that estimates the emission parameters from the training set using MLE (maximum likelihood estimation) (5 points)

# In[1]:


import os


# In[2]:


def estimate_emission_parameters_base(training_set):
    #Store the count for each state, count(y)
    state_count = {}
    #Store the state-observation count, count(y->x)
    state_observation_count = {}
    #Store the estimated emission parameters
    estimated_emission_parameters = {}
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
        else:
            continue
            
    
    #For each x|y, calculate count(y->x)/count(y)
    for k,v in state_observation_count.items():
        estimated_emission_parameters[k] = v/state_count[k[1]]
    
    return estimated_emission_parameters


# # Part 1.2: Write a function that estimates the emission parameters from the training set using MLE, accounting for words that appear in the test set that do not appear in the training set (maximum likelihood estimation) (10 points)

# In[3]:


def estimate_emission_parameters(training_set, k_value=1):
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
            
    #For each x|y, calculate count(y->x)/(count(y)+k) and calculate k/(count(y)+k) -> the x for this is #UNK#
    #We assume from any label y there is a certain chance of generating #UNK# as a rare event,
    #and emprically we assume we have observed that there are k occurences of such an event 
    for k,v in state_observation_count.items():
        estimated_emission_parameters[k] = v/(state_count[k[1]]+k_value)
        estimated_emission_parameters[("#UNK#",k[1])] = k_value/(state_count[k[1]]+k_value)
    
    return estimated_emission_parameters, list(trained_words)


# # Part 1.3: Implementation of simple sentiment analysis system (10 points)

# ## Train and Evaluate with ES

# #### Read ES Train Dataset

# In[4]:


filepath_ES_train = os.path.join(os.getcwd(), 'Data', 'ES', 'train')

#Read the file contents
with open(filepath_ES_train, 'r', encoding='utf-8') as file:
    file_contents_ES_train = file.readlines()
    
#Convert to training set
es_training_set = [w.strip() for w in file_contents_ES_train]


# #### Learn ES parameters

# In[5]:


#Calculate the parameters using the training set
all_estimated_emission_parameters, trained_words = estimate_emission_parameters(es_training_set)


# #### Learn ES parameters: Get argmax_y( e(x|y) )

# In[6]:


#Calculate y* = argmax_y e(x|y)
#i.e. find the y that produces the highest emission probability for x
estimated_emission_parameters = {}
for k,v in all_estimated_emission_parameters.items():
    #If the word is already in the estimated_emission_parameters
    if(k[0] in estimated_emission_parameters):
        #Check if its emission probability is greater than what has been stored previously
        #If it is greater, then update the tag and emission probability 
        if(v > estimated_emission_parameters[k[0]][1]):
            estimated_emission_parameters[k[0]] = [k[1],v]
    #else if the word is not already in estimated_emission_parameters
    #create an entry
    else:
        estimated_emission_parameters[k[0]] = [k[1],v]


# #### Read ES dev.in Dataset

# In[7]:


filepath_ES_devin = os.path.join(os.getcwd(), 'Data', 'ES', 'dev.in')

#Read the file contents
with open(filepath_ES_devin, 'r', encoding='utf-8') as file:
    file_contents_ES_devin = file.readlines()
    
es_devin = [w.strip() for w in file_contents_ES_devin]


# #### Evaluate on ES dev.in

# In[8]:


for i in range(len(es_devin)):
    #If its not an empty line
    if(len(es_devin[i])!=0):
        #If the word can be found in our learned emission parameters, add the learned label
        if(es_devin[i] in estimated_emission_parameters.keys()):
            es_devin[i] = es_devin[i] + " " + estimated_emission_parameters[es_devin[i]][0]
        #else, use the label for unknown
        else:
            es_devin[i] = es_devin[i] + " " + estimated_emission_parameters["#UNK#"][0]


# #### Write to dev.p1.out

# In[9]:


filepath_dev_p1_out = os.path.join(os.getcwd(), 'Data', 'ES', 'dev.p1.out')


# In[10]:


with open(filepath_dev_p1_out, 'w', encoding='utf-8') as file:
    for line in es_devin:
        file.write(line + '\n')


# #### Compare dev.p1.out with dev.out for ES
python evalResult.py dev.out dev.p1.out#Entity in gold data: 229
#Entity in prediction: 1466

#Correct Entity : 178
Entity  precision: 0.1214
Entity  recall: 0.7773
Entity  F: 0.2100

#Correct Sentiment : 97
Sentiment  precision: 0.0662
Sentiment  recall: 0.4236
Sentiment  F: 0.1145
# ## Train and Evaluate with RU

# #### Read RU Train Dataset

# In[11]:


filepath_RU_train = os.path.join(os.getcwd(), 'Data', 'RU', 'train')

#Read the file contents
with open(filepath_RU_train, 'r', encoding='utf-8') as file:
    file_contents_RU_train = file.readlines()
    
#Convert to training set
ru_training_set = [w.strip() for w in file_contents_RU_train]


# #### Learn RU parameters

# In[12]:


#Calculate the parameters using the training set
all_estimated_emission_parameters, trained_words = estimate_emission_parameters(ru_training_set)


# #### Learn RU parameters: Get argmax_y( e(x|y) )

# In[13]:


#Calculate y* = argmax_y e(x|y)
#i.e. find the y that produces the highest emission probability for x
estimated_emission_parameters = {}
for k,v in all_estimated_emission_parameters.items():
    #If the word is already in the estimated_emission_parameters
    if(k[0] in estimated_emission_parameters):
        #Check if its emission probability is greater than what has been stored previously
        #If it is greater, then update the tag and emission probability 
        if(v > estimated_emission_parameters[k[0]][1]):
            estimated_emission_parameters[k[0]] = [k[1],v]
    #else if the word is not already in estimated_emission_parameters
    #create an entry
    else:
        estimated_emission_parameters[k[0]] = [k[1],v]


# #### Read RU dev.in Dataset

# In[14]:


filepath_RU_devin = os.path.join(os.getcwd(), 'Data', 'RU', 'dev.in')

#Read the file contents
with open(filepath_RU_devin, 'r', encoding='utf-8') as file:
    file_contents_RU_devin = file.readlines()
    
ru_devin = [w.strip() for w in file_contents_RU_devin]


# #### Evaluate on RU dev.in

# In[15]:


for i in range(len(ru_devin)):
    #If its not an empty line
    if(len(ru_devin[i])!=0):
        #If the word can be found in our learned emission parameters, add the learned label
        if(ru_devin[i] in estimated_emission_parameters.keys()):
            ru_devin[i] = ru_devin[i] + " " + estimated_emission_parameters[ru_devin[i]][0]
        #else, use the label for unknown
        else:
            ru_devin[i] = ru_devin[i] + " " + estimated_emission_parameters["#UNK#"][0]


# #### Write to dev.p1.out

# In[16]:


filepath_dev_p1_out = os.path.join(os.getcwd(), 'Data', 'RU', 'dev.p1.out')


# In[17]:


with open(filepath_dev_p1_out, 'w', encoding='utf-8') as file:
    for line in ru_devin:
        file.write(line + '\n')


# #### Compare dev.p1.out with dev.out for RU
python evalResult.py dev.out dev.p1.out#Entity in gold data: 389
#Entity in prediction: 1816

#Correct Entity : 266
Entity  precision: 0.1465
Entity  recall: 0.6838
Entity  F: 0.2413

#Correct Sentiment : 129
Sentiment  precision: 0.0710
Sentiment  recall: 0.3316
Sentiment  F: 0.1170
# In[ ]:




