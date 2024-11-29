# gemma2-challenge
Gemma2-challenge for kaggle competition

# environment setup
run "generate_requirements.sh" to generate requirements.txt automatically, by scanning the code
optionally, first edit generate_requirements.sh to add more packages 'by hand'
run "pip install -r requirements.txt" to install all the required packages

copy .gitignore.template to .gitignore and modify it as needed
copy settings.template.py to settings.py and enter the values of the variables that will be used

# code overview
src/ contains the original training code by Filip and Miha
data_tools contains tools and utils for creating and manipulating the datasets
utils package is for various utility modules not located in more specific packages 

## training
training package contains the training code
full_training_v1.py is a refactored and slightly updated version of DPmultiGPU.py
