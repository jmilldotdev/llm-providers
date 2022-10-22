from typing import Any
from typing import Dict
from typing import List

from llm_providers.models import PreparedRequest
from llm_providers.provider import LLMProvider

OPENAI_ENGINES = {
    "text-davinci-002",
    "text-davinci-001",
    "davinci",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "code-davinci-002",
    "code-cushman-001",
}

# User param -> (client param, default value)
OPENAI_PARAMS = {
    "model": ("model", "text-davinci-002"),
    "temperature": ("temperature", 1.0),
    "max_tokens": ("max_tokens", 10),
    "n": ("n", 1),
    "top_p": ("top_p", 1.0),
    "logprobs": ("logprobs", None),
    "top_k_return": ("best_of", 1),
    "stop_sequence": ("stop", None),
    "presence_penalty": ("presence_penalty", 0.0),
    "frequency_penalty": ("frequency_penalty", 0.0),
}


class OpenAIProvider(LLMProvider):
    def connect(
        self,
        connection_str: str,
        client_args: Dict[str, Any] = {},
    ) -> None:
        self.api_key = connection_str
        self.base_url = "https://api.openai.com/v1"
        for key in OPENAI_PARAMS:
            setattr(self, key, client_args.pop(key, OPENAI_PARAMS[key][1]))
        if getattr(self, "model") not in OPENAI_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {OPENAI_ENGINES}.",
            )

    def get_model_params(self) -> Dict:
        return {"model_name": "openai", "model": getattr(self, "model")}

    def get_model_inputs(self) -> List:
        return list(OPENAI_PARAMS.keys())

    def prepare_request(
        self,
        prompt: str,
        request_args: Dict[str, Any] = {},
    ) -> PreparedRequest:
        api_url = self.base_url + "/completions"
        request_params = {"prompt": prompt}
        for key in OPENAI_PARAMS:
            request_params[OPENAI_PARAMS[key][0]] = request_args.pop(
                key,
                getattr(self, key),
            )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return PreparedRequest(api_url=api_url, params=request_params, headers=headers)

    def parse_completion(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["text"]
