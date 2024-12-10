"""Script for metrics.

More info:
- Machine translation metrics:
  - https://machinetranslate.org/metrics
"""
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from nltk.translate.chrf_score import corpus_chrf
from nltk.translate.nist_score import corpus_nist
from nltk.translate.ribes_score import corpus_ribes
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer

def corpus_meteor():
    pass

def _nltk_score_wrapper(nltk_corpus_score):
    def func(predictions, references):
        return nltk_corpus_score(references, predictions)
    return func

METRICS = {
    'bleu':  _nltk_score_wrapper(corpus_bleu),
    'gleu':  _nltk_score_wrapper(corpus_gleu),
    'chrf':  _nltk_score_wrapper(corpus_chrf),
    'nist':  _nltk_score_wrapper(corpus_nist),
    'ribes': _nltk_score_wrapper(corpus_ribes),
}

def get_metric(metric_name: str):
    if metric_name.lower() in METRICS:
        return METRICS[metric_name.lower()]
    else:
        raise NotImplementedError(f"The given metric ({metric_name}) is not implemented jet. " +
                                  f"The only implemented metrics are: ({list(METRICS.keys())})")