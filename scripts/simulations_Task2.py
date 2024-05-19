import openai
import transformers
import requests
import time
import random
import numpy as np
import re
from tqdm import tqdm
from itertools import combinations

# retry decorator (https://cookbook.openai.com/examples/how_to_handle_rate_limits)
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError, KeyError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

class VandenPsychMedTask_Standard:
    
    def __init__(self, model_name, trials_per_context=30, client=None, hf_token=None):
        self.model_name = model_name
        self.client = client
        self.hf_token = hf_token
        
        self.contexts = np.repeat(range(2), trials_per_context)                 
        self.options = np.array(['A','B','C','D'])              
        self.payoffs = np.zeros((2, 2, trials_per_context))  

        probs = np.array([[0.1, 0.4], [0.6, 0.9]])
        T = trials_per_context
        for i in range(2):
            for j in range(2):
                self.payoffs[i, j] = np.array([1] * round(probs[i, j] * T) + [0] * round((1 - probs[i, j]) * T))
                np.random.shuffle(self.payoffs[i, j])
        self.context_trial = [0, 0]

        np.random.shuffle(self.options)                                   
        self.options = self.options.reshape((2,2))                   
        trial_order = np.random.permutation(range(len(self.contexts)))              
        self.contexts = self.contexts[trial_order]
        
        self.learning_pairs = self.options[self.contexts]                                   
        self.transfer_pairs = np.array(list(combinations(self.options.flatten(), 2)))  
        self.option_indices = dict(zip(self.options.flatten(), range(4)))
        
        self.start_text = f"You are playing a game that involves choosing between different slot machines.\n"\
        "Each slot machine gives 1 point with a particular probability, otherwise 0 points.\n"\
        "Some slot machines have a higher probability of reward than others.\n"\
        "The goal is to maximize your total payoff over the course of several rounds.\n"\
        "Your total payoff will be the cumulative sum of the points you win across all rounds of the game.\n\n"
        
    def get_outcome_history(self):
        self.prompt = self.start_text
        if len(self.previous_feedback) > 0:
            self.prompt += "You made the following observations in the past:\n\n"
            for i, feedback in enumerate(self.previous_feedback):
                self.prompt += f"- In Round {i + 1}, " + feedback
                
    def choice_prompt(self, options):
        text = f"\nYou now face a choice between slot machine {options[0]} and slot machine {options[1]}.\n"\
        "Your goal is to maximize your payoffs.\n"\
        "Which slot machine do you choose? Give your answer like this: I would choose slot machine _. Do not explain why."
        return text
    
    # a model will occasionally give an aberrant response that doesn't fit the expected format
    # in this case, re.search() will return None 
    def recode_accuracy(self, response, correct_choice):
        choice = re.search("I would choose slot machine .", response)
        if choice is None:
            return -99
        else:
            choice = choice[0][-1]
            return 1 if choice == correct_choice else 0

    def recode_left(self, response, left_choice):
        choice = re.search("I would choose slot machine .", response)
        if choice is None:
            return -99
        else:
            choice = choice[0][-1]
            return 1 if choice == left_choice else 0
                    
    def draw_outcomes(self, context_id, trial_num):
        outcomes = self.payoffs[context_id, :, trial_num]
        return outcomes
    
    def append_feedback(self, options, outcomes, order):
        points = ["point" if outcomes[order[0]] == 1 else "points",
                  "point" if outcomes[order[1]] == 1 else "points"]
        feedback = f"slot machine {options[0]} delivered {int(outcomes[order[0]])} {points[0]} and slot machine {options[1]} delivered {int(outcomes[order[1]])} {points[1]}.\n"
        self.previous_feedback.append(feedback)
     
    def learning_phase_choice(self, trial_id):
        # add outcome history to prompt 
        self.get_outcome_history()
        
        # randomize order of the choice options in the prompt
        pair = self.learning_pairs[trial_id].copy()
        order = np.random.permutation(2)
        pair = pair[order]
        
        # add choice instructions to prompt
        self.prompt += self.choice_prompt(pair)

        #print(self.prompt)
        
        # send request to model
        response = self.query_model(self.prompt)

        #print(response)
        #print()
        
        # recode model response
        accuracy = self.recode_accuracy(response, self.learning_pairs[trial_id, 1])
        left = self.recode_left(response, pair[0])
        
        # draw outcomes and append to feedback list
        context = self.contexts[trial_id]
        outcomes = self.draw_outcomes(context, self.context_trial[context])
        self.context_trial[context] += 1
        self.append_feedback(pair, outcomes, order)
        
        # trial data
        L_idx = self.option_indices[self.learning_pairs[trial_id, 0]]
        H_idx = self.option_indices[self.learning_pairs[trial_id, 1]] 
        left_idx = self.option_indices[pair[0]]
        right_idx = self.option_indices[pair[1]]
        data = [trial_id, context, L_idx, H_idx, self.learning_pairs[trial_id, 0], self.learning_pairs[trial_id, 1], 
                left_idx, right_idx, pair[0], pair[1], response, accuracy, left, outcomes[0], outcomes[1]]
        
        return data
        
    def transfer_phase_choice(self, trial_id, order):
        # add outcome history to prompt 
        self.get_outcome_history()
        
        # set the order of the choice options in the prompt
        pair = self.transfer_pairs[trial_id].copy()
        pair = pair[order]
        
        # add choice instructions to prompt
        self.prompt += self.choice_prompt(pair)

        #print(self.prompt)
        
        # send request to model
        response = self.query_model(self.prompt)

        #print(response)
        #print()
        
        # recode model response
        accuracy = self.recode_accuracy(response, self.transfer_pairs[trial_id, 1])
        left = self.recode_left(response, pair[0])
        
        # trial data
        L_idx = self.option_indices[self.transfer_pairs[trial_id, 0]]
        H_idx = self.option_indices[self.transfer_pairs[trial_id, 1]]
        left_idx = self.option_indices[pair[0]]
        right_idx = self.option_indices[pair[1]]
        data = [trial_id, 99, L_idx, H_idx, self.transfer_pairs[trial_id, 0], self.transfer_pairs[trial_id, 1],
                left_idx, right_idx, pair[0], pair[1], response, accuracy, left, 0, 0]
        
        return data
        
    def simulate(self):
        self.previous_feedback = []
        self.data = []
        for trial in tqdm(range(len(self.learning_pairs)), desc="learning phase"):
            self.data.append(self.learning_phase_choice(trial))
        for trial in tqdm(range(len(self.transfer_pairs)), desc="transfer phase 1"):
            self.data.append(self.transfer_phase_choice(trial, order=np.array([0,1])))
        for trial in tqdm(range(len(self.transfer_pairs)), desc="transfer phase 2"):
            self.data.append(self.transfer_phase_choice(trial, order=np.array([1,0])))
        return self.data
    
    @retry_with_exponential_backoff
    def query_model(self, prompt, temperature=0, system="You are a helpful assistant."):
        if self.model_name.startswith('gpt'):
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                temperature = temperature)
            return response.choices[0].message.content.strip()
        elif self.model_name.startswith('meta-llama') or self.model_name.startswith('mistralai'):
            API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {"inputs": prompt, 
            "parameters": {"do_sample": True if temperature > 0 else False},
            "options": {"use_cache": False, "wait_for_model": True}}  #don't use cache- we want a fresh response
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()[0]['generated_text'][len(prompt):].strip()


class VandenPsychMedTask_Comparisons(VandenPsychMedTask_Standard):
    def append_feedback(self, options, outcomes, order):
        points = ["point" if outcomes[order[0]] == 1 else "points",
                  "point" if outcomes[order[1]] == 1 else "points"]
        diff = abs(outcomes[0] - outcomes[1])
        direction = "more" if outcomes[order[0]] >= outcomes[order[1]] else "less"
        diff_pts = "point" if diff == 1 else "points"
        feedback = f"slot machine {options[0]} delivered {int(outcomes[order[0]])} {points[0]}, which is {int(diff)} {diff_pts} {direction} than slot machine {options[1]} delivered ({int(outcomes[order[1]])} {points[1]}).\n"
        self.previous_feedback.append(feedback)