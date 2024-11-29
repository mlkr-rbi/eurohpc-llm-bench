'''
Utility classes for creating translation prompts for LLM training and querying.
'''

class TranslationPromptComposer:
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

    def _assign_lang_labels(self, start_lang: str) -> tuple[str, str]:
        ''' Create the 2 language labels in the appropriate order, based on the start language. '''
        if start_lang == self._lang1:
            label1, label2 = self._lang1_label, self._lang2_label
        elif start_lang == self._lang2:
            label1, label2 = self._lang2_label, self._lang1_label
        else:
            raise ValueError(f"start_lang must be either {self._lang1} or {self._lang2}")
        return label1, label2

    def train_prompt(self, text1: str, text2: str, start_lang: str) -> str:
        label1, label2 = self._assign_lang_labels(start_lang)
        instr = self._instruction + " " if self._instruction else ""
        return f"{instr}{label1}: {text1} {label2}: {text2}"

    def test_prompt(self, text: str, start_lang: str = None) -> str:
        label1, label2 = self._assign_lang_labels(start_lang)
        instr = self._instruction + " " if self._instruction else ""
        return f"{instr}{label1}: {text} {label2}: "

def hr_en_translate_prompt():
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

def prompt_tst():
    p = hr_en_translate_prompt()
    print(p.train_prompt('Dobar dan', 'Good day', 'hr'))
    print(p.test_prompt('Dobar dan', 'hr'))
    print(p.train_prompt('Good day', 'Dobar dan', 'en'))
    print(p.test_prompt('Good day', 'en'))

if __name__ == '__main__':
    prompt_tst()