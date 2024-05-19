import numpy as np

class Model:

    '''Basic delta RL model with unbiased outcome encoding and position bias.

    Parameters: [0] Learning rate
                [1] Inverse temperature
                [2] Position bias'''

    def __init__(self, params=None, Q_init=None):
        self.params = params
        self.Q_init = Q_init
        # set Rmin to +inf and Rmax to -inf 
        # this will ensure they update correctly on the first trial 
        self.r_min = np.Inf 
        self.r_max = -np.Inf

    # function for loading data into the model
    def set_data(self, choices, option_ids, outcomes, n_learning, n_options):
        self.choices = choices
        self.option_ids = option_ids
        self.outcomes = outcomes
        self.n_learning = n_learning
        self.n_options = n_options

    # subjective value function
    def value_function(self, x):
        x = x.astype(float)
        self.r_min = min(self.r_min, x.min()) # update Rmin
        self.r_max = max(self.r_max, x.max()) # update Rmax
        norm = self.r_max - self.r_min        # range 
        v = (x - self.r_min) / (norm if norm > 0 else 1.)
        return v

    # learning function
    def update_function(self, Q, v):
        return Q + self.params[0] * (v - Q)

    # (log) softmax
    def softmax(self, Q, log=False):
        position = np.zeros(len(Q))
        position[0] = 1.
        V = Q * self.params[1] + position * self.params[2]
        e_x = np.exp(V - np.max(V))
        if log:
            return V - np.max(V) - np.log(e_x.sum())
        else:
            return e_x / e_x.sum()

    # negative log-likelihood function
    # for fitting a single set of parameters to data pooled across multiple simulation runs 
    def objective(self, params):
        if params is not None:
            self.params = params
        n_runs, n_trials = self.choices.shape  
        LL = np.full((n_runs, n_trials), np.nan)
        for run in range(n_runs):
            # initialize Q values
            Q_values = np.full(self.n_options, self.Q_init, dtype=float)
            for trial in range(n_trials):
                ids = self.option_ids[run, trial]
                if self.choices[run, trial] != -99:
                    log_probs = self.softmax(Q_values[ids[ids >= 0]], log=True)
                    LL[run, trial] = log_probs[self.choices[run, trial]]
                if trial < self.n_learning:
                    v = self.value_function(self.outcomes[run, trial][ids >= 0])
                    Q_values[ids[ids >= 0]] = self.update_function(Q_values[ids[ids >= 0]], v)
        return -1 * LL[~np.isnan(LL)].sum()

    # predicted choice probability function
    # for generating predictions using a single set of parameters across multiple simulation runs
    def predict(self, params):
        if params is not None:
            self.params = params
        n_runs, n_trials = self.option_ids.shape[:2]
        probs = np.zeros(self.option_ids.shape)
        for run in range(n_runs):
            # initialize Q values
            Q_values = np.full(self.n_options, self.Q_init, dtype=float)
            for trial in range(n_trials):
                ids = self.option_ids[run, trial]
                Q_vals = np.array([Q_values[id] if id >= 0 else -np.inf for id in ids])
                probs[run, trial] = self.softmax(Q_vals)
                if trial < self.n_learning:
                    v = self.value_function(self.outcomes[run, trial][ids >= 0])
                    Q_values[ids[ids >= 0]] = self.update_function(Q_values[ids[ids >= 0]], v)
        return probs


class SeparateInvTempsModel(Model):

    '''Basic delta RL model with unbiased outcome encoding, position bias, and 
    separate inverse temperature parameters for the learning and transfer phases.

    Parameters: [0] Learning rate
                [1] Learning phase inverse temperature
                [2] Transfer phase inverse temperature
                [3] Position bias'''

    def softmax(self, Q, phase=1, log=False):
        position = np.zeros(len(Q))
        position[0] = 1.
        V = Q * self.params[phase] + position * self.params[3]
        e_x = np.exp(V - np.max(V))
        if log:
            return V - np.max(V) - np.log(e_x.sum())
        else:
            return e_x / e_x.sum()

    def objective(self, params):
        if params is not None:
            self.params = params
        n_runs, n_trials = self.choices.shape  
        LL = np.full((n_runs, n_trials), np.nan)
        for run in range(n_runs):
            # initialize Q values
            Q_values = np.full(self.n_options, self.Q_init, dtype=float)
            for trial in range(n_trials):
                ids = self.option_ids[run, trial]
                if trial < self.n_learning:
                    if self.choices[run, trial] != -99:
                        log_probs = self.softmax(Q_values[ids[ids >= 0]], phase=1, log=True)
                        LL[run, trial] = log_probs[self.choices[run, trial]]
                    v = self.value_function(self.outcomes[run, trial][ids >= 0])
                    Q_values[ids[ids >= 0]] = self.update_function(Q_values[ids[ids >= 0]], v)
                else:
                    if self.choices[run, trial] != -99:
                        log_probs = self.softmax(Q_values[ids[ids >= 0]], phase=2, log=True)
                        LL[run, trial] = log_probs[self.choices[run, trial]]
        return -1 * LL[~np.isnan(LL)].sum()

    def predict(self, params):
        if params is not None:
            self.params = params
        n_runs, n_trials = self.option_ids.shape[:2]
        probs = np.zeros(self.option_ids.shape)
        for run in range(n_runs):
            # initialize Q values
            Q_values = np.full(self.n_options, self.Q_init, dtype=float)
            for trial in range(n_trials):
                ids = self.option_ids[run, trial]
                Q_vals = np.array([Q_values[id] if id >= 0 else -np.inf for id in ids])
                if trial < self.n_learning:
                    probs[run, trial] = self.softmax(Q_vals, phase=1)
                    v = self.value_function(self.outcomes[run, trial][ids >= 0])
                    Q_values[ids[ids >= 0]] = self.update_function(Q_values[ids[ids >= 0]], v)
                else:
                    probs[run, trial] = self.softmax(Q_vals, phase=2)
        return probs


class ConfirmationBiasModel(Model):

    '''Basic delta RL model with unbiased outcome encoding, position bias, and 
    separate learning rates for confirmatory and disconfirmatory prediction errors.
    
    Parameters: [0] CON learning rate
                [1] DIS learning rate
                [2] Inverse temperature
                [3] Position bias'''

    def update_function(self, Q, v, choice):
        pred_errors = v - Q
        learn_rates = [0.] * len(Q)
        for i in range(len(Q)):
            if choice == i and pred_errors[i] >= 0:
                learn_rates[i] = self.params[0]
            elif choice == i and pred_errors[i] < 0:
                learn_rates[i] = self.params[1]
            elif choice != i and pred_errors[i] >= 0:
                learn_rates[i] = self.params[1]
            elif choice != i and pred_errors[i] < 0:
                learn_rates[i] = self.params[0]
            else:
                learn_rates[i] = (self.params[0] + self.params[1]) / 2.0

        return Q + learn_rates * pred_errors

    def softmax(self, Q, log=False):
        position = np.zeros(len(Q))
        position[0] = 1.
        V = Q * self.params[2] + position * self.params[3]
        e_x = np.exp(V - np.max(V))
        if log:
            return V - np.max(V) - np.log(e_x.sum())
        else:
            return e_x / e_x.sum()

    def objective(self, params):
        if params is not None:
            self.params = params
        n_runs, n_trials = self.choices.shape  
        LL = np.full((n_runs, n_trials), np.nan)
        for run in range(n_runs):
            # initialize Q values
            Q_values = np.full(self.n_options, self.Q_init, dtype=float)
            for trial in range(n_trials):
                ids = self.option_ids[run, trial]
                if self.choices[run, trial] != -99:
                    log_probs = self.softmax(Q_values[ids[ids >= 0]], log=True)
                    LL[run, trial] = log_probs[self.choices[run, trial]]
                if trial < self.n_learning:
                    v = self.value_function(self.outcomes[run, trial][ids >= 0])
                    Q_values[ids[ids >= 0]] = self.update_function(Q_values[ids[ids >= 0]], v, self.choices[run, trial])
        return -1 * LL[~np.isnan(LL)].sum()

    def predict(self, params):
        if params is not None:
            self.params = params
        n_runs, n_trials = self.option_ids.shape[:2]
        probs = np.zeros(self.option_ids.shape)
        for run in range(n_runs):
            # initialize Q values
            Q_values = np.full(self.n_options, self.Q_init, dtype=float)
            for trial in range(n_trials):
                ids = self.option_ids[run, trial]
                Q_vals = np.array([Q_values[id] if id >= 0 else -np.inf for id in ids])
                probs[run, trial] = self.softmax(Q_vals)
                if trial < self.n_learning:
                    v = self.value_function(self.outcomes[run, trial][ids >= 0])
                    Q_values[ids[ids >= 0]] = self.update_function(Q_values[ids[ids >= 0]], v, self.choices[run, trial])
        return probs


class ConfirmationBiasSITModel(SeparateInvTempsModel):

    '''Basic delta RL model with unbiased outcome encoding, position bias,
    separate learning rates for confirmatory and disconfirmatory prediction errors,
    and separate inverse temperature parameters for the learning and transfer phases.
    
    Parameters: [0] CON learning rate
                [1] DIS learning rate
                [2] Learning phase inverse temperature
                [3] Transfer phase inverse temperature
                [4] Position bias'''

    def update_function(self, Q, v, choice):
        pred_errors = v - Q
        learn_rates = [0.] * len(Q)
        for i in range(len(Q)):
            if choice == i and pred_errors[i] >= 0:
                learn_rates[i] = self.params[0]
            elif choice == i and pred_errors[i] < 0:
                learn_rates[i] = self.params[1]
            elif choice != i and pred_errors[i] >= 0:
                learn_rates[i] = self.params[1]
            elif choice != i and pred_errors[i] < 0:
                learn_rates[i] = self.params[0]
            else:
                learn_rates[i] = (self.params[0] + self.params[1]) / 2.0

        return Q + learn_rates * pred_errors

    def softmax(self, Q, phase=1, log=False):
        position = np.zeros(len(Q))
        position[0] = 1.
        V = Q * self.params[phase + 1] + position * self.params[4]
        e_x = np.exp(V - np.max(V))
        if log:
            return V - np.max(V) - np.log(e_x.sum())
        else:
            return e_x / e_x.sum()

    def objective(self, params):
        if params is not None:
            self.params = params
        n_runs, n_trials = self.choices.shape  
        LL = np.full((n_runs, n_trials), np.nan)
        for run in range(n_runs):
            # initialize Q values
            Q_values = np.full(self.n_options, self.Q_init, dtype=float)
            for trial in range(n_trials):
                ids = self.option_ids[run, trial]
                if trial < self.n_learning:
                    if self.choices[run, trial] != -99:
                        log_probs = self.softmax(Q_values[ids[ids >= 0]], phase=1, log=True)
                        LL[run, trial] = log_probs[self.choices[run, trial]]
                    v = self.value_function(self.outcomes[run, trial][ids >= 0])
                    Q_values[ids[ids >= 0]] = self.update_function(Q_values[ids[ids >= 0]], v, self.choices[run, trial])
                else:
                    if self.choices[run, trial] != -99:
                        log_probs = self.softmax(Q_values[ids[ids >= 0]], phase=2, log=True)
                        LL[run, trial] = log_probs[self.choices[run, trial]]
        return -1 * LL[~np.isnan(LL)].sum()

    def predict(self, params):
        if params is not None:
            self.params = params
        n_runs, n_trials = self.option_ids.shape[:2]
        probs = np.zeros(self.option_ids.shape)
        for run in range(n_runs):
            # initialize Q values
            Q_values = np.full(self.n_options, self.Q_init, dtype=float)
            for trial in range(n_trials):
                ids = self.option_ids[run, trial]
                Q_vals = np.array([Q_values[id] if id >= 0 else -np.inf for id in ids])
                if trial < self.n_learning:
                    probs[run, trial] = self.softmax(Q_vals, phase=1)
                    v = self.value_function(self.outcomes[run, trial][ids >= 0])
                    Q_values[ids[ids >= 0]] = self.update_function(Q_values[ids[ids >= 0]], v, self.choices[run, trial])
                else:
                    probs[run, trial] = self.softmax(Q_vals, phase=2)
        return probs

    
class RelativeModel(Model):

    '''Basic delta RL model with relative outcome encoding and position bias.

    Parameters: [0] Learning rate
                [1] Inverse temperature
                [2] Position bias
                [3] Relative encoding weight'''

    def value_function(self, x):
        x = x.astype(float)
        self.r_min = min(self.r_min, x.min()) # update Rmin
        self.r_max = max(self.r_max, x.max()) # update Rmax
        norm = self.r_max - self.r_min        # subjective range 
        x_abs = (x - self.r_min) / (norm if norm > 0 else 1.)
        if np.all(x == x[0]):
            x_rel = np.full(len(x), 1./len(x))
        else:
            x_rel = (x - np.min(x)) / (np.max(x) - np.min(x))
        v = (1 - self.params[3]) * x_abs + self.params[3] * x_rel
        return v


class RelativeSepInvTempsModel(SeparateInvTempsModel):

    '''Basic delta RL model with relative outcome encoding, position bias, and 
    separate inverse temperature parameters for the learning and transfer phases.

    Parameters: [0] Learning rate
                [1] Learning phase inverse temperature
                [2] Transfer phase inverse temperature
                [3] Position bias
                [4] Relative encoding weight'''

    def value_function(self, x):
        x = x.astype(float)
        self.r_min = min(self.r_min, x.min()) # update Rmin
        self.r_max = max(self.r_max, x.max()) # update Rmax
        norm = self.r_max - self.r_min        # subjective range 
        x_abs = (x - self.r_min) / (norm if norm > 0 else 1.)
        if np.all(x == x[0]):
            x_rel = np.full(len(x), 1./len(x))
        else:
            x_rel = (x - np.min(x)) / (np.max(x) - np.min(x))
        v = (1 - self.params[4]) * x_abs + self.params[4] * x_rel
        return v


class RelativeConfirmationBiasModel(ConfirmationBiasModel):

    '''Basic delta RL model with relative outcome encoding, position bias, and 
    separate learning rates for confirmatory and disconfirmatory prediction errors.
    
    Parameters: [0] CON learning rate
                [1] DIS learning rate
                [2] Inverse temperature
                [3] Position bias
                [4] Relative encoding weight'''

    def value_function(self, x):
        x = x.astype(float)
        self.r_min = min(self.r_min, x.min()) # update Rmin
        self.r_max = max(self.r_max, x.max()) # update Rmax
        norm = self.r_max - self.r_min        # subjective range 
        x_abs = (x - self.r_min) / (norm if norm > 0 else 1.)
        if np.all(x == x[0]):
            x_rel = np.full(len(x), 1./len(x))
        else:
            x_rel = (x - np.min(x)) / (np.max(x) - np.min(x))
        v = (1 - self.params[4]) * x_abs + self.params[4] * x_rel
        return v


class RelativeConfirmationBiasSITModel(ConfirmationBiasSITModel):

    '''Basic delta RL model with relative outcome encoding, position bias,
    separate learning rates for confirmatory and disconfirmatory prediction errors,
    and separate inverse temperature parameters for the learning and transfer phases.
    
    Parameters: [0] CON learning rate
                [1] DIS learning rate
                [2] Learning phase inverse temperature
                [3] Transfer phase inverse temperature
                [4] Position bias
                [5] Relative encoding weight'''

    def value_function(self, x):
        x = x.astype(float)
        self.r_min = min(self.r_min, x.min()) # update Rmin
        self.r_max = max(self.r_max, x.max()) # update Rmax
        norm = self.r_max - self.r_min        # subjective range 
        x_abs = (x - self.r_min) / (norm if norm > 0 else 1.)
        if np.all(x == x[0]):
            x_rel = np.full(len(x), 1./len(x))
        else:
            x_rel = (x - np.min(x)) / (np.max(x) - np.min(x))
        v = (1 - self.params[5]) * x_abs + self.params[5] * x_rel
        return v