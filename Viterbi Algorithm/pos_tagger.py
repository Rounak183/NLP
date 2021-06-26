import smoother as smoother
import n_gramer as n_gramer
import math
import numpy
from copy import deepcopy

def tag(words_to_tag,possible_tags,known_words,trigram_probs,emission_probs,start_token='*',stop_token='STOP'):
	
	tagged=[]

	# P(X|Y,Z)
	tags_transition_matrix=transition_probabilities_for_ngram_list(trigram_probs)

	# P(tag|word)
	possible_word_tags=transition_probabilities_for_ngram_list(emission_probs)

	possible_tags_states={}
	for tg in possible_tags:
		possible_tags_states[tg]=0.0	

	starting_probabilities=get_starting_probabilities(possible_tags_states,tags_transition_matrix)

	for word_list in words_to_tag:
		n_gramer.insert_start_end_tokens(word_list,start_token,stop_token,2)
		tagged_sentence=viterbi_path(word_list,possible_tags_states,known_words,tags_transition_matrix,possible_word_tags,starting_probabilities)
		tagged.append(tagged_sentence+'\n')

	return tagged

def viterbi_path(word_list_observations,possible_tags_states,known_words,tags_transition_matrix,emissions_matrix,starting_probabilities):
	path=[]
	vit=viterbi_matrix(word_list_observations,possible_tags_states,emissions_matrix)

	viterbi_matrix_2(vit)

	prior_state=starting_probabilities
	for i in range(2,len(word_list_observations)):
		prior_observation_words,current_obs_word=grams_from_list_at_index(i,word_list_observations)
		filtered_word=replace_rare_word_if_needed(known_words,current_obs_word)
		next_state=calculate_new_probabilities_for_state(prior_state,vit,filtered_word,i-1)

		predicted_tag=arg_max_from_state(next_state)
		prior_state=predicted_tag
		tg=''
		for k,v in predicted_tag.items():
			tg=k
		string=current_obs_word+'_'+tg
		path.append(string)

	path.pop()
	return ' '.join(path)

def arg_max_from_state(new_states):

	max_val=float('-inf')
	tg=''
	for k,v in new_states.items():
		if v>max_val and v!=0.0:
			max_val=v
			tg=k

	return {tg:max_val}

def viterbi_matrix(word_list_observations,possible_tags_states,emissions_matrix):
	matrix=[]

	for word in word_list_observations:
		tags=deepcopy(possible_tags_states)
		possible_hidden_states_for_word=transitions_for_prior((word, ),emissions_matrix)
		if possible_hidden_states_for_word==-1000:
			word='_RARE_'
			possible_hidden_states_for_word=transitions_for_prior((word, ),emissions_matrix)

		for k,v in possible_hidden_states_for_word.items():
			tags[k]=v
		matrix.append({word:tags})

	return matrix

def replace_rare_word_if_needed(known_words,word):
	if word in known_words:
		return word
	else:
		return '_RARE_'

def calculate_new_probabilities_for_state(prior_state,vit_matrix,word,index):
	next_state=vit_matrix[index][word]

	if prior_state==-1000:
		return next_state

	if len(prior_state)==1:
		for k,v in next_state.items():
			if k in prior_state:
				next_state[k]=v*prior_state[k]

		return next_state

	for k,v in prior_state.items():
		val=next_state[k]*v
		if val==-0.0:
			val=0
		next_state[k]=val

	return next_state

def viterbi_matrix_2(matrix):

	probability_calulcation=0
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if matrix[i][j]==1:
				probability_calulcation+=1

	calulcated_probabilities=probability_calulcation/(len(matrix)*len(matrix[0]))

def grams_from_list_at_index(i,word_list):

	word_tri=tuple(word_list[i-2:i])
	viterbi_matrix_2([[]])
	prior=word_tri[0:-1]
	current=word_tri[-1]
	return prior,current

def get_starting_probabilities(possible_tags_states,tags_transition_matrix):
	
	starting_probs=deepcopy(possible_tags_states)
	starting_transition_probabilites=transitions_for_prior(('*','*'),tags_transition_matrix)
	viterbi_matrix_2([[]])
	if (starting_transition_probabilites==-1000):
		return starting_transition_probabilites
	for k,v in starting_transition_probabilites.items():
		starting_probs[k]=v

	return starting_probs

def transitions_for_prior(prior,tags_transitions):
	
	if prior in tags_transitions:
		transitions=tags_transitions[prior]
	else:
		transitions=-1000
	return transitions

def transition_probabilities_for_ngram_list(ngrams):

	viterbi_matrix_2([[]])
	matrix={}
	for k,v in ngrams.items():
		prior=k[0:-1]
		current=k[-1]
		if prior in matrix:
			pairs_dict=matrix[prior]
		else:
			pairs_dict={}

		pairs_dict[current]=v
		matrix[prior]=pairs_dict

	return matrix

def emission_probabilities_from(words,tags):
	freqs=smoother.frequency_dict(tags)
	pair_freqs=pair_frequency(words,tags)
	emy=calculate_emissions(pair_freqs,freqs)

	known_tags=set(freqs.keys())
	return emy,known_tags

def pair_frequency(list_a,list_b):
	tuple_freq={}

	for word_list,tag_sentence in zip(list_a,list_b):
		tag_list=tag_sentence.strip().split()
		for word,tag in zip(word_list,tag_list):
			w_t_tuple=(word,tag)
			tuple_freq[w_t_tuple]=tuple_freq.get(w_t_tuple,0)+1

	return tuple_freq

def calculate_emissions(start_end_tuple_list,end_state_count_list):
	emissions={}

	for pair in start_end_tuple_list:
		word,tag=pair
		emission_probability=float(start_end_tuple_list[pair])/float(end_state_count_list[tag])

		emissions[pair]=math.log(emission_probability,2)

	viterbi_matrix_2([[]])
	return emissions

def possible_tags_for_word(input_word,tags):
	possible_tags={}
	most_probable_tag=''
	most_probable_tag_prob=float('-infinity')
	for k,v in tags.items():
		word,tg=k
		if word==input_word:
			possible_tags[tg]=v
			if v>most_probable_tag:
				most_probable_tag=tg
				most_probable_tag_prob=v

	return possible_tags,most_probable_tag
