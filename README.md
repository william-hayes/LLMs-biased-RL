# LLMs are Biased Reinforcement Learners
Code and data for "Large Language Models are Biased Reinforcement Learners."

## Abstract
In-context learning enables large language models (LLMs) to perform a variety of tasks, including learning to make reward-maximizing choices in simple bandit tasks. Given their potential use as (autonomous) decision-making agents, it is important to understand how these models perform such reinforcement learning (RL) tasks and the extent to which they are susceptible to biases. Motivated by the fact that, in humans, it has been widely documented that the value of an outcome depends on how it compares to other local outcomes, the present study focuses on whether similar value encoding biases apply to how LLMs encode rewarding outcomes. Results from experiments with multiple bandit tasks and models show that LLMs exhibit behavioral signatures of a relative value bias. Adding explicit outcome comparisons to the prompt produces opposing effects on performance, enhancing maximization in trained choice sets but impairing generalization to new choice sets. Computational cognitive modeling reveals that LLM behavior is well-described by a simple RL algorithm that incorporates relative values at the outcome encoding stage. Lastly, we present preliminary evidence that the observed biases are not limited to fine-tuned LLMs, and that relative value processing is detectable in the final hidden layer activations of a raw, pretrained model. These findings have important implications for the use of LLMs in decision-making applications.

## Instructions

1. The **scripts** folder contains code for reproducing the main analyses.

* The Jupyter notebooks with the [model]-[task]-[#] filenames can be used to run bandit task experiments with the LLMs.

* *RL_model_fitting.py* can be used to fit RL models to LLM choice data. It uses maximum likelihood estimation with multiple random start points. As an example, if you wanted to fit the ABS model (model 0) to the choices generated by llama-2-70b-chat in the B2018 task using the standard prompt, run the following from the command line:
```
python RL_model_fitting.py --dataset "../data/Task 1 (B2018)/modeling_data_Task1.csv" --condition "standard prompt" --agent "llama-2-70b-chat" --bandits 8 --trials 76 --learning 48 --model 0 --starts 100 --cores <n_cpu_cores>
```
* *Generate_Plots.ipynb* contains code for reproducing most of the plots in the paper.

* *Aggregate_Analyses.Rmd* runs the ANOVAs reported in the paper.

2. The **hidden_states** folder contains code and results for the analysis of hidden state activations in Gemma-7b. <ins>These analyses were carried out on a cloud A100-80G GPU.</ins>

* To reproduce the extraction of hidden activations, run the following from the command line (with condition = "standard" or "comparisons"):
```
python extraction.py --token <HF_API_token> --condition <prompt condition> --trials 10 --nruns 100 --start 0
```

* *hidden_states_analysis.ipynb* contains code for analyzing the extracted hidden activations.

3. The **data** folder contains the raw and aggregated data sets, as well as the RL modeling results.
