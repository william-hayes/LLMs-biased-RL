import openai
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

class BavardPalminteri2023Task_Standard:
    
    def __init__(self, model_name, trials_per_context=15, client=None, hf_token=None):
        self.model_name = model_name
        self.client = client
        self.hf_token = hf_token
        
        self.contexts = np.repeat(range(4), trials_per_context)                 
        self.options = np.array(['A','B','C','D','E','F','G','H','I','J'])              
        self.avg_payoffs = [np.array([14, 50]), np.array([14, 32, 50]), np.array([14, 86]), np.array([14, 50, 86])]

        np.random.shuffle(self.options)                                   
        self.options = np.split(self.options, [2, 5, 7]) # split into 4 contexts w/ sizes 2, 3, 2, and 3                  
        trial_order = np.random.permutation(range(len(self.contexts)))              
        self.contexts = self.contexts[trial_order]
        
        self.learning_sets = [self.options[c] for c in self.contexts]                                   
        self.transfer_pairs = np.array(list(combinations(np.concatenate(self.options), 2)))  
        self.option_indices = dict(zip(np.concatenate(self.options), range(10)))
        
        self.start_text = f"In this task, you will be given information about several slot machines in order to decide which ones you want to play.\n"\
        "Some slot machines win more money than others on average.\n"\
        "On each trial, you will be asked to choose between two or three different slot machines.\n"\
        "Your goal is to make choices that maximize your total payoffs. In other words, you should try to win as much money as possible.\n"\
        "Your total payoff will be the cumulative sum of the money you win across all rounds of the game.\n\n"
        
    def get_outcome_history(self):
        self.prompt = self.start_text
        if len(self.previous_feedback) > 0:
            self.prompt += "You made the following observations in the past:\n\n"
            for i, feedback in enumerate(self.previous_feedback):
                self.prompt += f"- In Round {i + 1}, " + feedback
                
    def choice_prompt(self, options):
        if len(options) == 2:
            text = f"\nYou now face a choice between slot machine {options[0]} and slot machine {options[1]}.\n"
        elif len(options) == 3:
            text = f"\nYou now face a choice between slot machine {options[0]}, slot machine {options[1]}, and slot machine {options[2]}.\n"
        text += "Your goal is to maximize your payoffs.\n"\
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
                    
    def draw_outcomes(self, context_id):
        outcomes = np.random.normal(loc=self.avg_payoffs[context_id],
                                    scale=2.0, 
                                    size=len(self.avg_payoffs[context_id])).astype(int)
        return outcomes
    
    def append_feedback(self, options, outcomes, order):
        if len(options) == 2:
            feedback = f"slot machine {options[0]} delivered {outcomes[order[0]]} dollars and slot machine {options[1]} delivered {outcomes[order[1]]} dollars.\n"
        elif len(options) == 3:
            feedback = f"slot machine {options[0]} delivered {outcomes[order[0]]} dollars, slot machine {options[1]} delivered {outcomes[order[1]]} dollars, and slot machine {options[2]} delivered {outcomes[order[2]]} dollars.\n"
        self.previous_feedback.append(feedback)
     
    def learning_phase_choice(self, trial_id):
        # add outcome history to prompt 
        self.get_outcome_history()
        
        # randomize order of the choice options in the prompt
        opts = self.learning_sets[trial_id].copy()
        order = np.random.permutation(len(opts))
        opts = opts[order]
        
        # add choice instructions to prompt
        self.prompt += self.choice_prompt(opts)

        #print(self.prompt)
        
        # send request to model
        response = self.query_model(self.prompt)

        #print(response)
        #print()
        
        # recode model response
        accuracy = self.recode_accuracy(response, self.learning_sets[trial_id][-1])
        left = self.recode_left(response, opts[0])
        
        # draw outcomes and append to feedback list
        outcomes = self.draw_outcomes(self.contexts[trial_id])
        self.append_feedback(opts, outcomes, order)
        
        # trial data
        if len(opts) == 2:
            L_idx = self.option_indices[self.learning_sets[trial_id][0]]
            M_idx = -99
            H_idx = self.option_indices[self.learning_sets[trial_id][1]]
            L_opt = self.learning_sets[trial_id][0]
            M_opt = "--"
            H_opt = self.learning_sets[trial_id][1]
            L_rew = outcomes[0]
            M_rew = 0
            H_rew = outcomes[1]
            left_idx = self.option_indices[opts[0]]
            middle_idx = -99
            right_idx = self.option_indices[opts[1]]
            left_opt = opts[0]
            middle_opt = "--"
            right_opt = opts[1]
        elif len(opts) == 3:
            L_idx = self.option_indices[self.learning_sets[trial_id][0]]
            M_idx = self.option_indices[self.learning_sets[trial_id][1]]
            H_idx = self.option_indices[self.learning_sets[trial_id][2]]
            L_opt = self.learning_sets[trial_id][0]
            M_opt = self.learning_sets[trial_id][1]
            H_opt = self.learning_sets[trial_id][2]
            L_rew = outcomes[0]
            M_rew = outcomes[1]
            H_rew = outcomes[2]
            left_idx = self.option_indices[opts[0]]
            middle_idx = self.option_indices[opts[1]]
            right_idx = self.option_indices[opts[2]]
            left_opt = opts[0]
            middle_opt = opts[1]
            right_opt = opts[2]
        data = [trial_id, self.contexts[trial_id], L_idx, M_idx, H_idx, L_opt, M_opt, H_opt,
                left_idx, middle_idx, right_idx, left_opt, middle_opt, right_opt, response, accuracy, left, L_rew, M_rew, H_rew]
        
        return data
        
    def transfer_phase_choice(self, trial_id):
        # add outcome history to prompt 
        self.get_outcome_history()
        
        # randomize order of the choice options in the prompt
        pair = self.transfer_pairs[trial_id].copy()
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
        accuracy = self.recode_accuracy(response, self.transfer_pairs[trial_id, 1])
        left = self.recode_left(response, pair[0])
        
        # trial data
        L_idx = self.option_indices[self.transfer_pairs[trial_id, 0]]
        H_idx = self.option_indices[self.transfer_pairs[trial_id, 1]]
        left_idx = self.option_indices[pair[0]]
        right_idx = self.option_indices[pair[1]]
        data = [trial_id, 99, L_idx, -99, H_idx, self.transfer_pairs[trial_id, 0], "--", self.transfer_pairs[trial_id, 1],
                left_idx, -99, right_idx, pair[0], "--", pair[1], response, accuracy, left, 0, 0, 0]
        
        return data
        
    def simulate(self):
        self.previous_feedback = []
        self.data = []
        for trial in tqdm(range(len(self.learning_sets)), desc="learning phase"):
            self.data.append(self.learning_phase_choice(trial))
        for trial in tqdm(range(len(self.transfer_pairs)), desc="transfer phase"):
            self.data.append(self.transfer_phase_choice(trial))
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


class BavardPalminteri2023Task_Comparisons(BavardPalminteri2023Task_Standard):
    def append_feedback(self, options, outcomes, order):
        if len(options) == 2:
            diff = abs(outcomes[0] - outcomes[1])
            direction = "more" if outcomes[order[0]] >= outcomes[order[1]] else "less"
            dollars = "dollar" if diff == 1 else "dollars"
            feedback = f"slot machine {options[0]} delivered {outcomes[order[0]]} dollars, which is {diff} {dollars} {direction} than slot machine {options[1]} delivered ({outcomes[order[1]]} dollars).\n"
        elif len(options) == 3:
            diff1 = abs(outcomes[order[0]] - outcomes[order[1]])
            diff2 = abs(outcomes[order[0]] - outcomes[order[2]])
            direction1 = "more" if outcomes[order[0]] >= outcomes[order[1]] else "less"
            direction2 = "more" if outcomes[order[0]] >= outcomes[order[2]] else "less"
            dollars1 = "dollar" if diff1 == 1 else "dollars"
            dollars2 = "dollar" if diff2 == 1 else "dollars"
            feedback = f"slot machine {options[0]} delivered {outcomes[order[0]]} dollars, which is {diff1} {dollars1} {direction1} than slot machine {options[1]} delivered ({outcomes[order[1]]} dollars), and {diff2} {dollars2} {direction2} than slot machine {options[2]} delivered ({outcomes[order[2]]} dollars).\n"
        self.previous_feedback.append(feedback)