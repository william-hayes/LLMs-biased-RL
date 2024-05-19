import transformers
import torch
import argparse 
import numpy as np
import pandas as pd
from extract_hidden_states import StateExtractor, StateExtractor_Comparisons

print(f'Using transformers version {transformers.__version__}')

parser = argparse.ArgumentParser()
parser.add_argument("--token", type=str, help="HF access token")
parser.add_argument("--condition", type=str, help="prompt condition")
parser.add_argument("--trials", type=int, help="learning trials per context")
parser.add_argument("--nruns", type=int, help="number of simulation runs")
parser.add_argument("--start", type=int, help="index of first run")
args = parser.parse_args()

# load tokenizer and model
model_ckpt = "google/gemma-7b"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt, token=args.token)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_ckpt, token=args.token, device_map="auto", attn_implementation="eager")

# data columns
columns = ['trial','context','L_index','H_index','L_option','H_option',
           'left_index','right_index','left_option','right_option',
           'choice','accuracy','chose_left','L_outcome','H_outcome']

# run the simulations
if __name__ == '__main__':
    if args.condition == "standard":
        for run in range(args.start, args.start + args.nruns):
            extractor = StateExtractor(trials_per_context=args.trials, model=model, tokenizer=tokenizer)
            print(f"\nRun {run}\n")
            data, hidden_states = extractor.simulate(get_choices=True)
            data = pd.DataFrame(data, columns=columns)
            data.to_csv(f"trial_data/standard {run}.csv", index=False)
            np.save(f"hidden_states/standard {run}.npy", hidden_states)
    elif args.condition == "comparisons":
        for run in range(args.start, args.start + args.nruns):
            extractor = StateExtractor_Comparisons(trials_per_context=args.trials, model=model, tokenizer=tokenizer)
            print(f"\nRun {run}\n")
            data, hidden_states = extractor.simulate(get_choices=True)
            data = pd.DataFrame(data, columns=columns)
            data.to_csv(f"trial_data/comparisons {run}.csv", index=False)
            np.save(f"hidden_states/comparisons {run}.npy", hidden_states)
