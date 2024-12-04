#!/bin/bash
# generic script for running a module via python commandline, in the background
# it adds . to pythonpath, redirects the output to file, and runs the command

# Check if a module path is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <module_path> <module_params>"
    exit 1
fi

# setup output file
MODULE_PATH="$1"
MODULE_NAME=$(basename "$MODULE_PATH" .py)
OUTPUT_FILE="${MODULE_NAME}.out.txt"

# take other parameters, if exist, and add as arguments to the module
ARGS=""
if [ "$#" -gt 1 ]; then
    shift
    ARGS="$@"
    echo "Arguments: $ARGS"
fi

# run in the background
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
nohup python "$MODULE_PATH" "$ARGS" > "$OUTPUT_FILE" 2>&1 &
echo "Running $MODULE_PATH in background. Output redirected to $OUTPUT_FILE"
