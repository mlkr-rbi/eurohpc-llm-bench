# generates requirements.txt using pipreqs tool

pip install pipreqs
pipreqs . --mode no-pin --savepath requirements.txt # remove no-pin to include the version numbers

# add additional "hidden" dependencies that pipreqs does not catch
deps="tf-keras nvidia-ml-py mpi4py deepspeed langchain-huggingface bitsandbytes tensorboard nltk rouge-score" # packages, space delimited

for dep in $deps
do
  echo $dep >> requirements.txt
done

# to install run:
# pip install -r requirements.txt