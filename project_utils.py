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