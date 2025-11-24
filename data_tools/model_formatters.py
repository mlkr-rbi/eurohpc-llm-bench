'''
Utility classes for wrapping translation prompts in model-specific chat/instruction templates.
Different LLMs have different prompt formats (Mistral, Llama, Gemma, etc.) that need to be
applied to the content-level translation instructions.
'''
from abc import ABC, abstractmethod
from typing import Optional


class ModelPromptFormatter(ABC):
    '''
    Abstract base class for model-specific prompt formatting.
    Wraps instruction content in model-specific template tokens and structure.
    '''

    @abstractmethod
    def format_train_prompt(self, instruction: str, response: str) -> str:
        '''
        Format instruction-response pair for training.
        :param instruction: The instruction text (e.g., translation task)
        :param response: The expected response/output
        :return: Formatted string with model-specific tokens
        '''
        pass

    @abstractmethod
    def format_inference_prompt(self, instruction: str) -> str:
        '''
        Format instruction for inference (without response).
        :param instruction: The instruction text
        :return: Formatted string with model-specific tokens
        '''
        pass


class IdentityFormatter(ModelPromptFormatter):
    '''
    Pass-through formatter that returns content unchanged.
    Used for backward compatibility and base models without special formatting.
    '''

    def format_train_prompt(self, instruction: str, response: str) -> str:
        return instruction

    def format_inference_prompt(self, instruction: str) -> str:
        return instruction


class MistralInstructFormatter(ModelPromptFormatter):
    '''
    Formatter for Mistral-7B-Instruct-v0.1 and similar Mistral instruct models.
    Format: <s>[INST] {instruction} [/INST]{response}</s>
    Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    '''

    def format_train_prompt(self, instruction: str, response: str) -> str:
        return f"<s>[INST] {instruction} [/INST]{response}</s>"

    def format_inference_prompt(self, instruction: str) -> str:
        return f"<s>[INST] {instruction} [/INST]"


class Llama2ChatFormatter(ModelPromptFormatter):
    '''
    Formatter for Llama-2-Chat models.
    Format: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]{response}</s>
    For translation tasks, we use a minimal system prompt.
    '''

    def __init__(self, system_prompt: str = "You are a helpful translation assistant."):
        self.system_prompt = system_prompt

    def format_train_prompt(self, instruction: str, response: str) -> str:
        return f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{instruction} [/INST]{response}</s>"

    def format_inference_prompt(self, instruction: str) -> str:
        return f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"


class GemmaInstructFormatter(ModelPromptFormatter):
    '''
    Formatter for Gemma instruction-tuned models.
    Format: <start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>
    '''

    def format_train_prompt(self, instruction: str, response: str) -> str:
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"

    def format_inference_prompt(self, instruction: str) -> str:
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def get_model_formatter(model_format: Optional[str] = None) -> ModelPromptFormatter:
    '''
    Factory function to get the appropriate formatter for a model.
    :param model_format: String identifier for the model format (e.g., 'mistral-instruct', 'llama2-chat')
    :return: ModelPromptFormatter instance
    '''
    formatters = {
        'mistral-instruct': MistralInstructFormatter(),
        'mistral-7b-instruct': MistralInstructFormatter(),
        'llama2-chat': Llama2ChatFormatter(),
        'gemma-it': GemmaInstructFormatter(),
        'gemma-instruct': GemmaInstructFormatter(),
    }

    if model_format is None:
        return IdentityFormatter()

    formatter = formatters.get(model_format.lower())
    if formatter is None:
        raise ValueError(f"Unknown model format: {model_format}. Available formats: {list(formatters.keys())}")

    return formatter


if __name__ == '__main__':
    # Test formatters
    instruction = "Translate the following text from English to Croatian.\nEnglish:\nHello world\n\nCroatian:\n"
    response = "Pozdrav svijetu"

    print("=== Identity Formatter ===")
    fmt = IdentityFormatter()
    print(fmt.format_train_prompt(instruction, response))
    print("\n" + "="*50 + "\n")

    print("=== Mistral Instruct Formatter ===")
    fmt = get_model_formatter('mistral-instruct')
    print(fmt.format_train_prompt(instruction, response))
    print("\nInference:")
    print(fmt.format_inference_prompt(instruction))
    print("\n" + "="*50 + "\n")

    print("=== Llama2 Chat Formatter ===")
    fmt = get_model_formatter('llama2-chat')
    print(fmt.format_train_prompt(instruction, response))
    print("\n" + "="*50 + "\n")

    print("=== Gemma Instruct Formatter ===")
    fmt = get_model_formatter('gemma-it')
    print(fmt.format_train_prompt(instruction, response))
