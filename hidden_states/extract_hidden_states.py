import transformers
import torch
import numpy as np
from tqdm import tqdm
from itertools import combinations


class StateExtractor:
    
    def __init__(self, trials_per_context, model, tokenizer):
        self.contexts = np.repeat(range(4), trials_per_context)                 
        self.options = np.array(['A','B','C','D','E','F','G','H'])              
        self.avg_payoffs = np.array([[15, 18], [21, 24], [27, 30], [33, 36]])   

        np.random.shuffle(self.options)                                   
        self.options = self.options.reshape((4,2))                   
        trial_order = np.random.permutation(range(len(self.contexts)))              
        self.contexts = self.contexts[trial_order]
        
        self.learning_pairs = self.options[self.contexts]                                   
        self.transfer_pairs = np.array(list(combinations(self.options.flatten(), 2)))  
        self.option_indices = dict(zip(self.options.flatten(), range(8)))
        
        self.start_text = f"You are playing a game with the goal of winning as much money as possible over the course of several rounds.\n"\
        "In each round, you will be asked which of two slot machines you wish to play.\n"\
        "Some slot machines win more money than others on average.\n"\
        f"Your total payoff will be the cumulative sum of the money you win across all rounds of the game.\n"\
        "Remember that your goal is to maximize your total payoff.\n\n"

        self.model = model
        self.tokenizer = tokenizer

    def get_outcome_history(self):
        self.prompt = self.start_text
        if len(self.previous_feedback) > 0:
            self.prompt += "You made the following observations in the past:\n\n"
            for i, feedback in enumerate(self.previous_feedback):
                self.prompt += f"- In Round {i + 1}, " + feedback
                
    def choice_prompt(self, options):
        text = f"\nYou now face a choice between slot machine {options[0]} and slot machine {options[1]}.\n"\
        "Your goal is to maximize your total payoff over the course of several rounds.\n\n"\
        "Q: Which slot machine do you choose?\n\n"\
        "A: I would choose slot machine"
        return text

    def recode_accuracy(self, response, correct_choice):
        if response is None:
            return -99
        else:
            return 1 if response == correct_choice else 0

    def recode_left(self, response, left_choice):
        if response is None:
            return -99
        else:
            return 1 if response == left_choice else 0
                
    def draw_outcomes(self, context_id):
        outcomes = np.random.normal(loc=self.avg_payoffs[context_id], scale=1.0, size=2).astype(int)
        return outcomes
    
    def append_feedback(self, options, outcomes, order):
        feedback = f"slot machine {options[0]} delivered {outcomes[order[0]]} dollars and slot machine {options[1]} delivered {outcomes[order[1]]} dollars.\n"
        self.previous_feedback.append(feedback)
     
    def learning_phase_trial(self, trial_id):
        # randomize order of the choice options in the prompt
        pair = self.learning_pairs[trial_id].copy()
        order = np.random.permutation(2)
        pair = pair[order]
        
        # draw outcomes and append to feedback list
        outcomes = self.draw_outcomes(self.contexts[trial_id])
        self.append_feedback(pair, outcomes, order)
        
        # trial data
        L_idx = self.option_indices[self.learning_pairs[trial_id, 0]]
        H_idx = self.option_indices[self.learning_pairs[trial_id, 1]]
        left_idx = self.option_indices[pair[0]]
        right_idx = self.option_indices[pair[1]]
        data = [trial_id, self.contexts[trial_id], L_idx, H_idx, self.learning_pairs[trial_id, 0], self.learning_pairs[trial_id, 1],
                left_idx, right_idx, pair[0], pair[1], '', -99, -99, outcomes[0], outcomes[1]]
        
        return data
        
    def transfer_phase_trial(self, trial_id, get_choice):
        # add outcome history to prompt 
        self.get_outcome_history()
        
        # randomize order of the choice options in the prompt
        pair = self.transfer_pairs[trial_id].copy()
        order = np.random.permutation(2)
        pair = pair[order]
        
        # add choice instructions to prompt
        self.prompt += self.choice_prompt(pair)

        # tokenize the prompt
        prompt_tokens = self.tokenizer(self.prompt, return_tensors='pt')

        # extract hidden states from the final layer for the last token in the prompt
        input_ids = prompt_tokens.input_ids.cuda()
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            final_layer_states = outputs.hidden_states[-1]
            last_token_states = final_layer_states[0, -1].cpu().numpy()
        self.hidden_states.append(last_token_states)

        # get choice from model
        if get_choice:
            prompt_length = input_ids.shape[-1]
            with torch.no_grad():
                outputs = self.model.generate(input_ids, do_sample=False, max_length=prompt_length + 1)
                response = self.tokenizer.decode(outputs[0][-1]).strip()
            # recode model response
            accuracy = self.recode_accuracy(response, self.transfer_pairs[trial_id, 1])
            left = self.recode_left(response, pair[0])
        else:
            response = ''
            accuracy = -99
            left = -99
        
        # trial data
        L_idx = self.option_indices[self.transfer_pairs[trial_id, 0]]
        H_idx = self.option_indices[self.transfer_pairs[trial_id, 1]]
        left_idx = self.option_indices[pair[0]]
        right_idx = self.option_indices[pair[1]]
        data = [trial_id, 99, L_idx, H_idx, self.transfer_pairs[trial_id, 0], self.transfer_pairs[trial_id, 1],
                left_idx, right_idx, pair[0], pair[1], response, accuracy, left, 0, 0]
        
        return data
        
    def simulate(self, get_choices=False):
        self.previous_feedback = []
        self.data = []
        self.hidden_states = []
        for trial in tqdm(range(len(self.learning_pairs)), desc="learning phase"):
            self.data.append(self.learning_phase_trial(trial))
        for trial in tqdm(range(len(self.transfer_pairs)), desc="transfer phase"):
            self.data.append(self.transfer_phase_trial(trial, get_choices))
        hidden_states = np.stack(self.hidden_states)
        return self.data, hidden_states


class StateExtractor_Comparisons(StateExtractor):
    def append_feedback(self, options, outcomes, order):
        diff = abs(outcomes[0] - outcomes[1])
        direction = "more" if outcomes[order[0]] >= outcomes[order[1]] else "less"
        dollars = "dollar" if diff == 1 else "dollars"
        feedback = f"slot machine {options[0]} delivered {outcomes[order[0]]} dollars, which is {diff} {dollars} {direction} than slot machine {options[1]} delivered ({outcomes[order[1]]} dollars).\n"
        self.previous_feedback.append(feedback)