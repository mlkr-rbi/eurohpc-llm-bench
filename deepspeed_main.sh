EXPERIMENT_FILE="mt-train-local-test-deepspeed.yml"
DEEPSPEED_CONFIG="experiments/deepspeed/deepspeed_test.json"

deepspeed main.py --deepspeed $DEEPSPEED_CONFIG --experiment $EXPERIMENT_FILE