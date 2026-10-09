import ollama


DEFAULT_HOST = "http://localhost:11434" # this is the standard host path, change as needed.
DEFAULT_MODEL = "qwen3:7B" #  "default model"
DEFAULT_TIMEOUT = 120 # timeout in seconds
DEFAULT_THINK = False
class OllamaClient:
    """
    A client interface that connects the ollama server to the code.
    """
    def __init__(self, 
                model:str = DEFAULT_MODEL,
                host:str = DEFAULT_HOST,
                timeout:str = DEFAULT_TIMEOUT,
                think:bool = DEFAULT_THINK
                ):
        self.model = model,
        self.think = think
        self.client = ollama.Client(host=host, timeout=timeout)
    
    def generate(self, 
                user_prompt:str, # pass in the user prompt containing details of the hypothesis + paper details
                system_prompt:str = None, # None is fine but make edits else it'll not work well
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
            options=options
        )
        return response["response"] 

    def stream(self, 
                user_prompt:str,
                system_prompt:str = None,
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

if __name__  == "__main__": # for debugging purposes only.
    client = OllamaClient()
    print(client.generate("Why is the meaning of life 42?"))