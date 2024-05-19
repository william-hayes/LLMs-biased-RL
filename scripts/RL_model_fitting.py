import argparse 
import numpy as np
import pandas as pd
from scipy.optimize import minimize 
from multiprocessing.pool import Pool
import multiprocessing as mp
#from tqdm import tqdm 

import RL_models

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="dataset for modeling")
parser.add_argument("--condition", type=str, help="which condition to model")
parser.add_argument("--agent", type=str, help="which LLM agent to model")
parser.add_argument("--bandits", type=int, help="total number of bandits")
parser.add_argument("--trials", type=int, help="total number of trials in task")
parser.add_argument("--learning", type=int, help="number of learning phase trials in task")
parser.add_argument("--model", type=int, help="index of RL model you want to fit")
parser.add_argument("--starts", type=int, help="number of random start points")
parser.add_argument("--cores", type=int, help="number of cores to use")
args = parser.parse_args()
n_options = args.bandits
n_trials = args.trials
n_learning = args.learning

# model fitting utilities
class FitModel:
    def __init__(self, objective_func, bounds):
        self.objective_func = objective_func
        self.bounds = bounds
    def __call__(self, x0):
        fit = minimize(self.objective_func, x0=x0, method='Nelder-Mead', bounds=self.bounds)
        return fit['fun'], fit['x']

if args.model == 0:
    model = RL_models.Model(Q_init = 0.5)
    bounds = [(.001, 0.999), (.001, 99.999), (-100, 100)]
elif args.model == 1:
    model = RL_models.SeparateInvTempsModel(Q_init = 0.5)
    bounds = [(.001, 0.999), (.001, 99.999), (.001, 99.999), (-100, 100)]
elif args.model == 2:
    model = RL_models.ConfirmationBiasModel(Q_init = 0.5)
    bounds = [(.001, 0.999), (.001, 0.999), (.001, 99.999), (-100, 100)]
elif args.model == 3:
    model = RL_models.ConfirmationBiasSITModel(Q_init = 0.5)
    bounds = [(.001, 0.999), (.001, 0.999), (.001, 99.999), (.001, 99.999), (-100, 100)]
elif args.model == 4:
    model = RL_models.RelativeModel(Q_init = 0.5)
    bounds = [(.001, 0.999), (.001, 99.999), (-100, 100), (0., 1.)]
elif args.model == 5:
    model = RL_models.RelativeSepInvTempsModel(Q_init = 0.5)
    bounds = [(.001, 0.999), (.001, 99.999), (.001, 99.999), (-100, 100), (0., 1.)]
elif args.model == 6:
    model = RL_models.RelativeConfirmationBiasModel(Q_init = 0.5)
    bounds = [(.001, 0.999), (.001, 0.999), (.001, 99.999), (-100, 100), (0., 1.)]
elif args.model == 7:
    model = RL_models.RelativeConfirmationBiasSITModel(Q_init = 0.5)
    bounds = [(.001, 0.999), (.001, 0.999), (.001, 99.999), (.001, 99.999), (-100, 100), (0., 1.)]

fit_func = FitModel(model.objective, bounds)

# load the data
data = pd.read_csv(args.dataset)

# subset the data for specified condition and agent
df = data.loc[(data['condition'] == args.condition) & (data['model'] == args.agent)]

# prepare data
ids = df['id'].value_counts(sort=False).index 
n_runs = len(ids)
if 'middle_index' in df.columns:
    option_cols = ['left_index', 'middle_index', 'right_index']
    outcome_cols = ['left_outcome', 'middle_outcome', 'right_outcome']
    k = 3
else:
    option_cols = ['left_index', 'right_index']
    outcome_cols = ['left_outcome', 'right_outcome']
    k = 2
option_ids = np.zeros((n_runs, n_trials, k), dtype=int)
choices = np.zeros((n_runs, n_trials), dtype=int)
outcomes = np.zeros((n_runs, n_trials, k), dtype=float)
for run in range(n_runs):
    option_ids[run] = df.loc[df['id'] == ids[run], option_cols].to_numpy(dtype=int)
    choices[run] = df.loc[df['id'] == ids[run], 'choice_idx'].to_numpy(dtype=int)
    outcomes[run] = df.loc[df['id'] == ids[run], outcome_cols].to_numpy(dtype=float)

# load data into models
model.set_data(choices, option_ids, outcomes, n_learning, n_options)

lb, ub = zip(*bounds)
start_points = [np.random.uniform(low=lb, high=ub) for i in range(args.starts)]

if __name__ == '__main__':
    # run minimizer with multiple random starting points in parallel
    #print(mp.cpu_count())
    print(f"Fitting model {args.model}...\n")
    pool = Pool(args.cores) 
    results = pool.map(fit_func, start_points)
    # collect results
    fn_vals, estimates = zip(*results)
    fn_vals = np.stack(fn_vals)
    estimates = np.stack(estimates)
    min_id = np.argmin(fn_vals)
    # calculate BIC
    bic = 2 * fn_vals[min_id] + len(bounds) * np.log(len(choices[choices != -99]))
    # compute predicted probabilities using fitted parameters
    preds = model.predict(estimates[min_id])
    # save results
    np.savez(f'{args.condition}_{args.agent}_model_{args.model}_results.npz', iter_values=fn_vals, iter_estimates=estimates, preds=preds)
    print(f"Best-fitting parameters: {np.around(estimates[min_id], 3)} (neg LL = {np.around(fn_vals[min_id], 3)}, BIC = {np.around(bic, 3)}).\n")
    pool.close()


