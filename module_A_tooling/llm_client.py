import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, Type

import requests
from pydantic import BaseModel

try: # only OllamaClient needs this, B2 works without it
    import ollama
except ImportError:
    ollama = None

from module_A_tooling.cache import Cache, get_default_cache, hash_text, make_key
from module_A_tooling.config import (CHARS_PER_TOKEN_ESTIMATE, LLM_MAX_ATTEMPTS, LLM_MODELS,
                                     LLM_TIMEOUT_SECONDS, RETRY_BACKOFF_FACTOR, RETRY_BASE_SECONDS)

_MISSING = object() # means "not in the cache"


# ---------------------------------------------------------------- errors

class LLMConfigError(Exception):
    """Server not running, wrong model name or bad setting. Retrying cannot help, so the run should stop."""

class LLMError(Exception):
    """This ONE question failed. Other questions can go on."""

class LLMTruncated(LLMError):
    """The answer was cut off at max_tokens."""

class LLMInputTooLong(LLMError):
    """The question does not fit into the model's context window."""


# ---------------------------------------------------------------- result

@dataclass
class LLMResult:
    data: dict # the checked answer
    model: str # nickname
    input_tokens: int
    output_tokens: int
    seconds: float
    cached: bool # True if the answer came from the cache
    attempts: int


# ---------------------------------------------------------------- talking to the local server

class LocalServer:
    """
    Sends ONE chat request to an OpenAI-compatible local server (Ollama, vLLM, LM Studio ...).
    Returns (text, input tokens, output tokens).
    """
    def __init__(self,
                timeout:float = LLM_TIMEOUT_SECONDS
                ):
        self.timeout = timeout

    def __call__(self, spec:dict, system:str, prompt:str, schema:Optional[dict]):
        body = {"model": spec["model"],
                "stream": False,
                "max_tokens": spec.get("max_tokens", 1500),
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}]}
        if spec.get("temperature") is not None:
            body["temperature"] = spec["temperature"]
        if spec.get("seed") is not None:
            body["seed"] = spec["seed"] # same question -> same answer (as far as the server allows)
        if spec.get("think") is False:
            body["reasoning_effort"] = "none" # asks the server to skip thinking
        mode = spec.get("structured_output", "json_schema")
        if mode == "json_schema" and schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": schema, "strict": True}}
        elif mode in ("json_schema", "json_object"):
            body["response_format"] = {"type": "json_object"}

        url = spec["base_url"].rstrip("/") + "/chat/completions"
        try:
            resp = requests.post(url, json=body, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise LLMConfigError(f"cannot reach the local LLM server at {spec['base_url']}: is it running?") from exc

        if resp.status_code >= 400:
            detail = (resp.text or "")[:300]
            if "context" in detail.lower():
                raise LLMInputTooLong(f"the server says the input is too long: {detail}")
            if resp.status_code in (400, 401, 403, 404, 422):
                raise LLMConfigError(f"the local server refused the request (HTTP {resp.status_code}): {detail}")
            raise RuntimeError(f"HTTP {resp.status_code}") # 429 / 5xx: busy, worth a retry

        data = resp.json()
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMTruncated(f"the answer was cut at max_tokens={body['max_tokens']}: raise max_tokens in config.LLM_MODELS")
        usage = data.get("usage") or {}
        text = (choice.get("message") or {}).get("content") or ""
        return text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def extract_json(text:str):
    """
    Finds the JSON object inside a model reply.
    Handles a ```json fence, a sentence around the object and the <think>...</think> block of some models.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1]) # a bad JSON raises ValueError too


# ---------------------------------------------------------------- the client used by B2

class LLMClient:
    """
    Asks the local LLM a question and returns a CHECKED answer (it always fits the pydantic class you pass).
    Same question to the same model is never asked twice (cache).

    client = LLMClient()
    res = client.call("b2a_review", SYSTEM, prompt, ReviewDraftB2a, model="local", version="1")
    res.data # dict that fits ReviewDraftB2a
    """
    def __init__(self,
                cache:Optional[Cache] = None,
                models:Optional[dict] = None, # nickname -> settings, default is config.LLM_MODELS
                provider:Optional[Callable] = None, # replace the server with a fake one in tests
                max_attempts:int = LLM_MAX_ATTEMPTS,
                sleep:Callable = time.sleep
                ):
        self.cache = cache if cache is not None else get_default_cache()
        self.models = models if models is not None else LLM_MODELS
        self.provider = provider if provider is not None else LocalServer()
        self.max_attempts = max_attempts
        self.sleep = sleep

    def spec(self, nickname:str):
        """Settings of one model nickname (from config.LLM_MODELS)."""
        spec = self.models.get(nickname)
        if spec is None:
            raise LLMConfigError(f"unknown model nickname {nickname!r}; known: {sorted(self.models)}")
        if str(spec.get("model", "")).startswith("SET_ME"):
            raise LLMConfigError(f"the model name of {nickname!r} is not set: edit config.LLM_MODELS")
        return spec

    def call(self,
            role:str, # a label, e.g. "b2a_review"
            system:str,
            prompt:str,
            schema:Type[BaseModel], # the pydantic class the answer must fit
            *,
            model:str, # nickname
            version:str = "1" # change it when the prompt changes, so old cached answers are not reused
            ):
        spec = self.spec(model)
        json_schema = schema.model_json_schema()

        # 1. cache: the key holds everything the answer depends on
        key = make_key(f"llm.{role}",
                       {"system": hash_text(system), "prompt": hash_text(prompt), "schema": json_schema,
                        "temperature": spec.get("temperature"),
                        "mode": spec.get("structured_output", "json_schema")},
                       model=spec["model"], revision=version)
        hit = self.cache.get(key, _MISSING)
        if hit is not _MISSING:
            return LLMResult(hit["data"], model, hit["input_tokens"], hit["output_tokens"],
                             hit["seconds"], True, hit["attempts"])

        # 2. refuse a question that is too long, instead of letting the server cut it silently
        context = spec.get("context_tokens")
        if context:
            needed = (len(system) + len(prompt)) / CHARS_PER_TOKEN_ESTIMATE + spec.get("max_tokens", 1500)
            if needed > context:
                raise LLMInputTooLong(f"{model}: about {int(needed)} tokens needed but the context is {context}. "
                                      "Shorten the input or raise context_tokens in config.LLM_MODELS")

        # 3. ask, check the answer, retry if needed
        feedback, last_error = "", None
        tokens_in = tokens_out = 0
        started = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            wait = RETRY_BASE_SECONDS * RETRY_BACKOFF_FACTOR ** (attempt - 1) # 1 s, 2 s, 4 s ...
            try:
                text, t_in, t_out = self.provider(spec, system, prompt + feedback, json_schema)
            except (LLMConfigError, LLMError):
                raise
            except Exception as exc: # timeout or busy server
                last_error = exc
            else:
                tokens_in += t_in
                tokens_out += t_out
                try:
                    data = schema.model_validate(extract_json(text)).model_dump()
                except ValueError as exc: # not JSON, or JSON that does not fit the class
                    last_error = exc
                    feedback = (f"\n\nYour previous reply could not be used ({str(exc)[:300]}). "
                                "Reply again with ONLY the JSON object, nothing else.")
                    wait = 0 # the model did answer, no need to wait
                else:
                    seconds = round(time.monotonic() - started, 3)
                    self.cache.set(key, {"data": data, "input_tokens": tokens_in, "output_tokens": tokens_out,
                                         "seconds": seconds, "attempts": attempt}, kind="llm")
                    return LLMResult(data, model, tokens_in, tokens_out, seconds, False, attempt)
            if attempt < self.max_attempts and wait > 0:
                self.sleep(wait)
        raise LLMError(f"{model}: failed after {self.max_attempts} attempts: {type(last_error).__name__}: {str(last_error)[:300]}")


# ---------------------------------------------------------------- OllamaClient (copied from local_llm_inference.py, nothing removed)

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

# if __name__  == "__main__": # for debugging purposes only.
#     client = OllamaClient()
#     print(client.generate("Why is the meaning of life 42?"))
    # for chunk in client.stream("Why is the meaning of life 42?"):
    #     print(chunk, end="", flush=True)
    # print()
