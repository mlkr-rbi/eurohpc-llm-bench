#!/bin/bash
# usage code2cluster.sh [machine name] [code folder] [preserve settings] [deploy folder]
# push local repo to a remote machine
# machine name can be 'mn' or 'leo', make sure to define the
#   correct SSH_HOST and REMOTE_ROOT for the machine below
# code folder is the path to the local repo, the last folder
#   in the path will be taken as the project name, and will be copied to remote machine
# deploy folder is the path where the project will be copied to on the remote machine
#   if not provided, it will be set to ~/code
# preserve settings (yes/no) will determine if the settings files on the remote machine are overwritten
#   use no for the first time, and if a complete project reset is needed
#   use yes when developing code, to keep the settings files on the remote machine
# EXAMPLE:
# position the terminal in the folder above the project folder (gemma2-challenge) with the code
# run: ./gemma2-challenge/hpc_tech/code2cluster.sh leo gemma2-challenge yes

## SETUP VARIABLES
# set SSH_HOST based on the machine name
if [ $# -lt 3 ]; then
    echo "Machine name, code folder, and preserve settings are required. Use 'mn' or 'leo' for machine name."
    exit 1
fi

MACHINE_NAME=$1
CODE_FOLDER=$2
PRESERVE_SETTINGS=$3
# check if PRESERVE_SETTINGS is 'yes' or 'no'
if [ "$PRESERVE_SETTINGS" != "yes" ] && [ "$PRESERVE_SETTINGS" != "no" ]; then
    echo "Invalid value for preserve settings. Use 'yes' or 'no'."
    exit 1
fi

# depending on the machine name, set:
# SSH_HOST - an alias in ~/.ssh/config or a full ssh address
# REMOTE_ROOT - the 'root' shared project folder where 'models', 'outputs', 'datasets' are stored
if [ "$MACHINE_NAME" == "mn" ]; then
    SSH_HOST="mna"
    REMOTE_ROOT="/gpfs/projects/ehpc124"
elif [ "$MACHINE_NAME" == "leo" ]; then
    SSH_HOST="leo" # placeholder for leo machine
    REMOTE_ROOT="/leonardo_work/EUHPC_B18_060"
else
    echo "Invalid machine name. Use 'mn' or 'leo'."
    exit 1
fi

# set PROJECT as the last folder in the path of the code folder
PROJECT=$(basename "$CODE_FOLDER")

DEPLOY_FOLDER="~/code" # where $PROJECT will be copied to remotely
# redefine DEPLOY_FOLDER if passed as argument
if [ $# -gt 3 ]; then
    DEPLOY_FOLDER=$4
fi

# delete remote project in the deploy folder, if exists
# delete only if PRESERVE_SETTINGS is 'no'
if [ "$PRESERVE_SETTINGS" == "no" ]; then
    echo 'Deleting remote project...'
    ssh $SSH_HOST "rm -rf $DEPLOY_FOLDER/$PROJECT"
else
    echo 'Preserving remote project settings...'
fi

echo 'Copying local project to remote machine...'
# do copy; add files and folders to the exclude list as needed
if [ "$PRESERVE_SETTINGS" == "no" ]; then
  rsync -av --exclude={'**/.git','**/training_output'} $CODE_FOLDER $SSH_HOST:$DEPLOY_FOLDER
else
  rsync -av --exclude={'**/.git','**/training_output'} \
       --filter='- *.json' \
       --filter='- *.yml' \
       --filter='- run.slurm' \
       --filter='- settings.py' \
       $CODE_FOLDER $SSH_HOST:$DEPLOY_FOLDER
fi

# create link-folders 'models', 'outputs', 'datasets' in the copied project
# pointing to the shared folders in $REMOTE_ROOT with the same names
for x in models outputs datasets; do
    echo "Creating link to $x..."
    # create link only if $DEPLOY_FOLDER/$PROJECT/$x does not exist
    ssh $SSH_HOST "[ ! -d $DEPLOY_FOLDER/$PROJECT/$x ] && ln -s $REMOTE_ROOT/$x $DEPLOY_FOLDER/$PROJECT/$x"
done

# create folder for slurm to store output, otherwise slurm run will fail
# this needs to be synced with run.slurm.X.template files, where output folder name is defined
ssh $SSH_HOST "[ ! -d $DEPLOY_FOLDER/$PROJECT/slurm ] && mkdir $DEPLOY_FOLDER/$PROJECT/slurm"

if [ "$PRESERVE_SETTINGS" == "yes" ]; then
    exit 0
fi
# create local versions of configuration files from their templates
# settings.py
echo 'initializing settings.py with a copy of settings.template.py...'
ssh $SSH_HOST "rm $DEPLOY_FOLDER/$PROJECT/settings.py"
ssh $SSH_HOST "cp $DEPLOY_FOLDER/$PROJECT/settings.template.py $DEPLOY_FOLDER/$PROJECT/settings.py"
# run.slurm
ssh $SSH_HOST "rm $DEPLOY_FOLDER/$PROJECT/run.slurm"
if [ "$MACHINE_NAME" == "mn" ]; then
    ssh $SSH_HOST "cp $DEPLOY_FOLDER/$PROJECT/run.slurm.mn.template $DEPLOY_FOLDER/$PROJECT/run.slurm"
elif [ "$MACHINE_NAME" == "leo" ]; then
    ssh $SSH_HOST "cp $DEPLOY_FOLDER/$PROJECT/run.slurm.leo.template $DEPLOY_FOLDER/$PROJECT/run.slurm"
fi
