# Codebase for the term paper "Knowledge Augmentation for Multi-Agent Orchestration"

## Reproducibility Guide
1. Clone the repository
2. Clone the REALM-Bench repository into a "third_party" folder within the main directory of the codebase
3. Install requirements, ideally in a virtual environment using 
´´pip install -r requirements.txt´´
4. Build the docker image for the code interpreter.

## Running the Experiments
Experiments can be run by executing the ´´run_experiments_fixed.py´´. The experiment parameters can be chosen using arguments:

- "--experiments": choose (mulitple) from the REALM-Bench scenarios
- "--seeds": chose a value for random seeds (default 42)
- "--model-config": choose the model provider for the experiment (default: openai, others are not working)
- "--verbose": choose to run verbose for extended debugging

## Disclaimer
The multi-agent setup stems from a private project and was built before this term paper. For more information on adaptations made to fit the context of the paper, see appendix of the paper.