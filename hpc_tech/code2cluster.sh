#!/bin/bash
# usage code2cluster.sh [machine name] [code folder] [deploy folder]
# push local repo to a remote machine
# machine name can be 'mn' or 'leo', make sure to define the
#   correct SSH_HOST and REMOTE_ROOT for the machine below
# code folder is the path to the local repo, the last folder
#   in the path will be taken as the project name, and will be copied to remote machine
# deploy folder is the path where the project will be copied to on the remote machine
#   if not provided, it will be set to ~/code

## SETUP VARIABLES
# set SSH_HOST based on the machine name
if [ $# -lt 2 ]; then
    echo "Machine name and code folder are required. Use 'mn' or 'leo' for machine name."
    exit 1
fi

MACHINE_NAME=$1
CODE_FOLDER=$2

# depending on the machine name, set:
# SSH_HOST - an alias in ~/.ssh/config or a full ssh address
# REMOTE_ROOT - the 'root' shared project folder where 'models', 'outputs', 'datasets' are stored
if [ "$MACHINE_NAME" == "mn" ]; then
    SSH_HOST="mn"
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
if [ $# -gt 2 ]; then
    DEPLOY_FOLDER=$3
fi

# delete remote project in the deploy folder, if exists
echo 'Deleting remote project...'
ssh $SSH_HOST "rm -rf $DEPLOY_FOLDER/$PROJECT"

echo 'Copying local project to remote machine...'
# do copy; add files and folders to the exclude list as needed
rsync -av --exclude={'**/.git','**/training_output'} $CODE_FOLDER $SSH_HOST:$DEPLOY_FOLDER

# create link-folders 'models', 'outputs', 'datasets' in the copied project
# pointing to the shared folders in $REMOTE_ROOT with the same names
for x in models outputs datasets; do
    echo "Creating link to $x..."
    ssh $SSH_HOST "ln -s $REMOTE_ROOT/$x $DEPLOY_FOLDER/$PROJECT/$x"
done

echo 'initializing settings.py with a copy of settings.template.py...'
ssh $SSH_HOST "rm $DEPLOY_FOLDER/$PROJECT/settings.py"
ssh $SSH_HOST "cp $DEPLOY_FOLDER/$PROJECT/settings.template.py $DEPLOY_FOLDER/$PROJECT/settings.py"
