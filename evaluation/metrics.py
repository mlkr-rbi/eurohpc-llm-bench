from evaluate import load
from functools import partial

METRICS = {
    'bleu':   partial(load, 'bleu'), 
    # 'bleurt':   partial(load, 'bleurt'), 
    'meteor': partial(load, 'meteor'), 
    # 'nist':   partial(load, 'nist'), 
    # 'chrf':   partial(load, 'chrf'), 
    # 'ter':    partial(load, 'ter'),
    'rouge':  partial(load, 'rouge'),
}

def get_metric(metric_name: str):
    if metric_name.lower() in METRICS:
        return METRICS[metric_name.lower()]()
    else:
        raise NotImplementedError(f"The given metric ({metric_name}) is not implemented jet. " +
                                  f"The only implemented metrics are: ({list(METRICS.keys())})")