'''
Utility classes for creating translation prompts for LLM training and querying.
'''
from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
from utils import config_utils


class TranslationPromptComposerABC(ABC):
    '''
    Utility class that transforms parallel texts to and from a string prompt for LLM training and querying.
    Prompts are based on a translation instruction, and language labels, and are of the form:
    INSTRUCTION LABEL1: text1 LABEL2: text2
    '''
    @abstractmethod
    def train_prompt(self, text1: str, text2: str, start_lang: str) -> str:
        return ""
    
    @abstractmethod
    def query_prompt(self, text: str, start_lang: str) -> str:
        return ""


class TranslationPromptComposer(TranslationPromptComposerABC):
    '''
    Utility class that transforms parallel texts to and from a string prompt for LLM training and querying.
    Prompts are based on a translation instruction, and language labels, and are of the form:
    INSTRUCTION LABEL1: text1 LABEL2: text2
    '''

    def __init__(self, lang1: str, lang2: str, lang1_label: None, lang2_label: None, instruction: str = ""):
        '''
        :param lang1: language 1 ISO 639-1 code (e.g. "en")
        :param lang2: language 2 ISO 639-1 code (e.g. "hr")
        :param lang1_label: label for language 1 (e.g. "English")
        :param lang2_label: label for language 2 (e.g. "Croatian")
        :param instruction: instruction for the translation task (e.g. "Translate the following text:")
        '''
        self._lang1, self._lang2 = lang1, lang2
        self._lang1_label = lang1_label if lang1_label else lang1
        self._lang2_label = lang2_label if lang2_label else lang2
        self._instruction = instruction

    def _assign_lang_labels(self, start_lang: str) -> Tuple[str, str]:
        ''' Create the 2 language labels in the appropriate order, based on the start language. '''
        if start_lang == self._lang1:
            label1, label2 = self._lang1_label, self._lang2_label
        elif start_lang == self._lang2:
            label1, label2 = self._lang2_label, self._lang1_label
        else:
            raise ValueError(f"start_lang must be either {self._lang1} or {self._lang2}")
        return label1, label2

    def train_prompt(self, text1: str, text2: str, start_lang: str) -> str:
        return f"{self.query_prompt(text1, start_lang)}{text2}"

    def query_prompt(self, text: str, start_lang: str) -> str:
        label1, label2 = self._assign_lang_labels(start_lang)
        instr = self._instruction + " " if self._instruction else ""
        return f"{instr}{label1}: {text} {label2}: "


class TranslationPromptComposerFromConfig(TranslationPromptComposerABC):
    '''
    Utility class that transforms parallel texts to and from a string prompt for LLM training and querying.
    Prompts are based on a translation instruction, and language labels, and are of the form:
    INSTRUCTION LABEL1: text1 LABEL2: text2
    '''

    def __init__(self, prompt_config: str, instruct_lang: str,
                 randomize_prompts: bool = False, seed: int = 123, instruction_tune: bool = False):
        '''
        :param prompt_config: prompt configuration file name")
        :param instruction_tune: patch to properly handle query formatting for instruction-tune prompts
        # TODO: handle instruction-tune prompt composition in a more general way
        '''
        self.prompt_config = config_utils.get_prompts(prompt_config, instruct_lang)
        self.start_langs = list(self.prompt_config.keys())
        self.randomize_prompts = randomize_prompts
        self.rng = np.random.default_rng(seed)
        self.idx = -1
        self.instruction_tune = instruction_tune
        
    def get_next_instruction(self, start_lang: str) -> str:
        prompts = self.prompt_config[start_lang]
        if self.randomize_prompts:
            self.idx = self.rng.integers(0, len(prompts))
        else:
            self.idx = (self.idx + 1) % len(prompts)
        return prompts[self.idx]

    def train_prompt(self, text1: str, text2: str, start_lang: str) -> str:
        return self.get_next_instruction(start_lang).format(input=text1, output=text2)

    def query_prompt(self, text1: str, start_lang: str) -> str:
        prompt = self.get_next_instruction(start_lang).format(input=text1, output="")
        if not self.instruction_tune: return prompt
        else: # remove '<end_of_turn>\n' from the prompt
            assert prompt.endswith('<end_of_turn>\n')
            return prompt[:-len('<end_of_turn>\n')]

def get_prompt(prompt_config: str,
               instruct_lang: str,
               randomize_prompts: bool = False,
               instruction_tune: bool = False) -> TranslationPromptComposerABC:
    '''
    Factory method for the default translation prompt composer.
    '''
    return TranslationPromptComposerFromConfig(
        prompt_config=prompt_config,
        instruct_lang=instruct_lang,
        randomize_prompts=randomize_prompts,
        instruction_tune=instruction_tune)


def hr_en_translate_prompt() -> TranslationPromptComposerABC:
    '''
    Factory method for the default Croatian-English translation prompt composer.
    '''
    return TranslationPromptComposer(
        lang1="hr",
        lang2="en",
        lang1_label="CROATIAN",
        lang2_label="ENGLISH",
        instruction="TRANSLATE:"
    )

def hr_en_translate_prompt_parabstract() -> TranslationPromptComposerABC:
    return TranslationPromptComposerFromConfig(
        prompt_config="mt-en-hr-single-instruct",
        instruct_lang="en",
        randomize_prompts=False,
        instruction_tune=False
    )

def prompt_tst():
    p = hr_en_translate_prompt()
    print(p.train_prompt('Dobar dan', 'Good day', 'hr'))
    print(p.query_prompt('Dobar dan', 'hr'))
    print(p.train_prompt('Good day', 'Dobar dan', 'en'))
    print(p.query_prompt('Good day', 'en'))


def prompt_tst2():
    p = get_prompt("mt-en-hr-001", "en", randomize_prompts=True)
    print("--------------------------------------------")
    print(p.train_prompt('Dobar dan', 'Good day', 'hr'))
    print("--------------------------------------------")
    print(p.query_prompt('Dobar dan', 'hr'))
    print("--------------------------------------------")
    print(p.train_prompt('Good day', 'Dobar dan', 'en'))
    print("--------------------------------------------")
    print(p.query_prompt('Good day', 'en'))
    print("--------------------------------------------")
    p = get_prompt("mt-en-hr-001", "hr")
    print("--------------------------------------------")
    print(p.train_prompt('Dobar dan', 'Good day', 'hr'))
    print("--------------------------------------------")
    print(p.query_prompt('Dobar dan', 'hr'))
    print("--------------------------------------------")
    print(p.train_prompt('Good day', 'Dobar dan', 'en'))
    print("--------------------------------------------")
    print(p.query_prompt('Good day', 'en'))


def prompt_tst3():
    p = get_prompt("mt-en-hr-002", "en-hr", randomize_prompts=True)
    

if __name__ == '__main__':
    prompt_tst()
    prompt_tst2()