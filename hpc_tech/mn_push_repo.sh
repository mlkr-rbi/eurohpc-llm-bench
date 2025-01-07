#!/bin/bash
# usage mn_push_repo [code folder] [deploy folder] [ssh host]
# push local repo to a remote machine
# code and deploy folders are PARENT folders of the project with code
# which is assumed to be named 'gemma2-challenge'
PROJECT="gemma2-challenge"

CODE_FOLDER="." # where PROJECT is located locally
# redefine CODE_FOLDER if passed as argument
if [ $# -gt 0 ]; then
    CODE_FOLDER=$1
fi
DEPLOY_FOLDER="~/code" # where PROJECT will be copied to remotely
# redefine DEPLOY_FOLDER if passed as argument
if [ $# -gt 1 ]; then
    DEPLOY_FOLDER=$2
fi
SSH_HOST="mna" # put full username@URL if no alias is configured in .ssh/config
# redefine SSH_HOST if passed as argument
if [ $# -gt 2 ]; then
    SSH_HOST=$3
fi

# delete remote project in the deploy folder, if exists
echo 'Deleting remote project...'
ssh $SSH_HOST "rm -rf $DEPLOY_FOLDER/$PROJECT"

echo 'Copying local project to remote machine...'
# do copy; add files and folders to the exclude list as needed
rsync -av --exclude={'**/.git','**/training_output'} $CODE_FOLDER/$PROJECT $SSH_HOST:$DEPLOY_FOLDER

# create link-folders 'models', 'outputs', 'datasets' in the copied project
ROOT_DIR="/gpfs/projects/ehpc124"
# create links from $CODE_FOLDER/$PROJECT/x to $ROOT_DIR/x  where x is 'models', 'outputs', 'datasets'
for x in models outputs datasets; do
    echo "Creating link to $x..."
    ssh $SSH_HOST "ln -s $ROOT_DIR/$x $DEPLOY_FOLDER/$PROJECT/$x"
done

echo 'initializing settings.py with a copy of settings.template.py...'
ssh $SSH_HOST "rm $DEPLOY_FOLDER/$PROJECT/settings.py"
ssh $SSH_HOST "cp $DEPLOY_FOLDER/$PROJECT/settings.template.py $DEPLOY_FOLDER/$PROJECT/settings.py"
