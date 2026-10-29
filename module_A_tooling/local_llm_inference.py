from typing import Optional, TypeVar

import ollama
from pydantic import BaseModel, ValidationError


DEFAULT_HOST = "http://localhost:11434" # this is the standard host path, change as needed.
DEFAULT_MODEL = "qwen3:8b" #  "default model"
DEFAULT_TIMEOUT = 120 # timeout in seconds
DEFAULT_THINK = True
class OllamaClient:
    """
    A client interface that connects the ollama server to the code.
    """
    def __init__(self, 
                model:str = DEFAULT_MODEL,
                host:str = DEFAULT_HOST,
                timeout:int = DEFAULT_TIMEOUT,
                think:bool = DEFAULT_THINK
                ):
        self.model = model
        self.think = think
        self.client = ollama.Client(host=host, timeout=timeout)
    
    def generate(self, 
                user_prompt:str, # pass in the user prompt containing details of the hypothesis + paper details
                system_prompt:Optional[str] = None, # None is fine but make edits else it'll not work well
                format=None, # optional JSON schema (dict) or "json" to force structured output
                **options
                ):
        """
        Returns the generated output in a single pass once complete.
        """
        response = self.client.generate(
            model=self.model, 
            prompt=user_prompt, 
            system=system_prompt,
            think=self.think,
            format=format,
            options=options
        )
        return response["response"] 

    def stream(self, 
                user_prompt:str,
                system_prompt:Optional[str] = None,
                **options
                ):
        """
        Returns the generated output in a stream. Exactly the same as generate
        """
        for chunk in self.client.generate(model=self.model, 
                                          prompt=user_prompt, 
                                          system=system_prompt, 
                                          options=options,
                                          think=self.think,
                                          stream=True):
            yield chunk["response"]

    def chat(self, messages, **options):
        response = self.client.generate(
                    model=self.model, 
                    messages=messages,
                    think=self.think, 
                    options=options
                )
        """
        Returns the generated output in message format, ie with LLM and user turns. 
        Makes it easy for role identification.
        """
        return response["messages"]["content"] 


T = TypeVar("T", bound=BaseModel)


def llm_structured(prompt: str,
                   schema: type[T],
                   client: OllamaClient,
                   system: Optional[str] = None,
                   retries: int = 2,
                   **options) -> T:
    """
    Asks the model for JSON matching `schema` and returns the validated object.
    Retries `retries` extra times on invalid output, then raises the last error.
    """
    last_error = None
    for _ in range(retries + 1):
        text = client.generate(user_prompt=prompt, system_prompt=system,
                               format=schema.model_json_schema(), **options)
        try:
            return schema.model_validate_json(text)
        except ValidationError as e:
            last_error = e
    raise last_error

# if __name__  == "__main__": # for debugging purposes only.
#     client = OllamaClient()
#     print(client.generate("Why is the meaning of life 42?"))
    # for chunk in client.stream("Why is the meaning of life 42?"):
    #     print(chunk, end="", flush=True)
    # print()