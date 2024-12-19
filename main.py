import argparse
from evaluation import evaluation
from training import full_training_v2, full_training_v3
from utils import config_utils

ACTIONS = {
    'training': full_training_v3,
    'evaluation': evaluation,
}

def get_parser():
    parser = argparse.ArgumentParser(
        prog='cro-gemma',
        description='The program for training and evaluating LLMs for Croatian language.',
        )
    parser.add_argument("--experiment",
                        help="Path to the experiment config YAML file.",
                        required=True,
                        type=str)
    parser.add_argument("--action",
                        help="Path to the experiment config YAML file.",
                        choices=ACTIONS.keys(),
                        required=False,
                        type=str)
    return parser


def main():
    kwargs = vars(get_parser().parse_known_args()[0])
    _kwargs = config_utils.get_experiment_arguments(**kwargs)
    action = _kwargs['action'].lower()
    if action in ACTIONS:
        kwargs = vars(ACTIONS[action].get_parser().parse_args())
        ACTIONS[action].main(**kwargs)


if __name__=="__main__":
    main()