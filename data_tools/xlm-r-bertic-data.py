'''
Code from: https://huggingface.co/datasets/classla/xlm-r-bertic-data/resolve/main/xlm-r-bertic-data.py
An example of how to compose a hf dataset from multiple sources (text-per-line gzipped files)
'''

import datasets
import gzip
import os
from typing import List


_URL = "http://nl.ijs.si/nikola/dedup_hbs/"
_URLS = {
    "macocu_hbs": _URL + "macocu.hbs.translit.dedup.lines.gz",
    "hr_news": _URL + "hr_news.translit.dedup.lines.gz",
    "bswac": _URL + "bswac.translit.dedup.lines.gz",
    "cc100_hr": _URL + "cc100-hr.translit.dedup.lines.gz",
    "cc100_sr": _URL + "cc100-sr.translit.dedup.lines.gz",
    "classla_sr": _URL + "classla-sr.translit.dedup.lines.gz",
    "classla_hr": _URL + "classla-hr.translit.dedup.lines.gz",
    "classla_bs": _URL + "classla-bs.translit.dedup.lines.gz",
    "cnrwac": _URL + "cnrwac.translit.dedup.lines.gz",
    "hrwac": _URL + "hrwac.translit.dedup.lines.gz",
    "mC4": _URL + "mC4.sr.translit.dedup.lines.gz",
    "riznica": _URL + "riznica.translit.dedup.lines.gz",
    "srwac": _URL + "srwac.translit.dedup.lines.gz",
}

_HOMEPAGE = _URL

_DESCRIPTION = """\
Data used to train XLM-Roberta-Bertić.
"""
_CITATION = r"""
To be added soon."""


class BerticData(datasets.GeneratorBasedBuilder):
    """Bertic dataset, used for training Bertic model."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from BerticDataConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=url,
                gen_kwargs={"filepath": downloaded_files[url]},
            )
            for url in _URLS
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        key = 0
        for name in [filepath]:
            # with gzip.open(name, "rb") as f:
            with open(name, "r") as f:
                for line in f.readlines():
                    yield key, {"text": line.rstrip()}
                    key += 1
