from typing import Any
from typing import Dict
from typing import List

from llm_providers.models import PreparedRequest
from llm_providers.provider import LLMProvider

COHERE_MODELS = {"small", "medium", "large", "xlarge"}

COHERE_PARAMS = {
    "model": ("model", "xlarge"),
    "max_tokens": ("max_tokens", 20),
    "temperature": ("temperature", 0.75),
    "num_generations": ("num_generations", 1),
    "k": ("k", 0),
    "p": ("p", 0.75),
    "frequency_penalty": ("frequency_penalty", 0.0),
    "presence_penalty": ("presence_penalty", 0.0),
    "stop_sequences": ("stop_sequences", []),
    "return_likelihoods": ("return_likelihoods", ""),
    "logit_bias": ("logit_bias", {}),
}


class CohereProvider(LLMProvider):
    def connect(
        self,
        connection_str: str,
        client_args: Dict[str, Any] = {},
    ) -> None:
        self.api_key = connection_str
        self.base_url = "https://api.cohere.ai"
        for key in COHERE_PARAMS:
            setattr(self, key, client_args.pop(key, COHERE_PARAMS[key][1]))
        if getattr(self, "model") not in COHERE_MODELS:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {COHERE_MODELS}.",
            )

    def get_model_params(self) -> Dict:
        return {"model_name": "openai", "model": getattr(self, "model")}

    def get_model_inputs(self) -> List:
        return list(COHERE_PARAMS.keys())

    def prepare_request(
        self,
        query: str,
        request_args: Dict[str, Any] = {},
    ) -> PreparedRequest:
        api_url = self.base_url + "/generate"
        request_params = {"prompt": query}
        for key in COHERE_PARAMS:
            request_params[COHERE_PARAMS[key][0]] = request_args.pop(
                key,
                getattr(self, key),
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Cohere-Version": "2021-11-08",
        }
        return PreparedRequest(api_url=api_url, params=request_params, headers=headers)

    def parse_completion(self, response: Dict[str, Any]) -> str:
        return response["generations"][0]["text"]
