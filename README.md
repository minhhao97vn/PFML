# Poisoning Attacks on Fair Machine Learning

This code replicates the experiments from the paper "Poisoning Attacks on Fair Machine Learning" submitted to AAAI-22

## Dependencies
In this project, we use Python 3.7.9 and dependencies on win-64 system.
 - Install python 3.7.9: https://www.python.org/downloads/release/python-379/
 - Install pip for python: https://pip.pypa.io/en/stable/installation/#
 - Install dependencies: `pip install -r pip-requirement.txt`

Note: In order to avoid conflicts between the dependencies from other projects, we highly recommend using python virtual environment. The details can be found here: https://docs.python.org/3/tutorial/venv.html

## Source code
### Structure
To work on our source code, you might want to modify the following files:
 - `utils.py`
   Utility functions to generate data, compute loss, gradient and fairness gap.
 - `train.py`
   Functions to train fair machine learning models (Reduction and Post-processing).
 - `attacks.py`
   Implementation of the PFML algorithm
 - `run_params_pretraining.py`

   Implementation to pretrain model's parameters. 

   To run the pretrain, use terminal with these arguments:

   `--lmd`: Lambda parameter

   `--learning_rate`: learning rate parameter

   `--fair_measure`: fairness measurement (use `equalized_odds` or `demographic_parity`)

   `--dataset`: data set (use `compas` or `adult`)

 - `run_pfml_attack.py`

   Implementation to run the PFML attack. 

   To run the PFML attack, use terminal with these arguments:

   `--epsilon`: fraction of poisoning samples/clean data (epsilon <= 1.0)

   `--lmd`: Lambda parameter

   `--gamma`: Gamma parameter

   `--learning_rate`: learning rate parameter

   `--flip_label`: use this argument for adversarial label flipping

   `--flip_feature`: use this argument for adversarial feature modification

   `--fair_measure`: fairness measurement (use `equalized_odds` or `demographic_parity`)

   `--dataset`: data set (use `compas` or `adult`)

### Run the code
Use `run_params_pretraining.py` to get pretrained parameters (pseudo code line 1-4)
 - To pretrain model's parameters, execute `python run_params_pretraining.py --lmd 5 --learning_rate 0.001 --dataset compas --fair_measure equalized_odds --random_state 1711`

Use `run_pfml_attack.py` to run the PFML attack (pseudo code line 5-10)
 - To run adversarial sampling, execute `python run_pfml_attack.py --epsilon 0.1 --lmd 5 --gamma 150 --learning_rate 0.001 --fair_measure equalized_odds --dataset compas --random_state 1`
 - To run adversarial label flipping, execute `python run_pfml_attack.py --epsilon 0.1 --lmd 5 --gamma 150 --learning_rate 0.001 --fair_measure equalized_odds --dataset compas --flip_label --random_state 1`
 - To run adversarial feature modification, execute `python run_pfml_attack.py --epsilon 0.1 --lmd 5 --gamma 150 --learning_rate 0.001 --fair_measure equalized_odds --dataset compas --flip_feature --random_state 1`
  
   Consider changing the arguments to try different settings. The results will be written into csv files.

   For reproducibility, we randomly pick random states in the range [0, 10000] to run experiments and collect results.

Note: To implement and get the baselines for this paper, some parts of our code are from other projects:
 - [Stronger Data Poisoning Attacks Break Data Sanitization Defenses](https://github.com/kohpangwei/data-poisoning-journal-release)
 - [On Adversarial Bias and the Robustness of Fair Machine Learning](https://github.com/privacytrustlab/adversarial_bias)
 - [Fairlearn](https://github.com/fairlearn/fairlearn)
