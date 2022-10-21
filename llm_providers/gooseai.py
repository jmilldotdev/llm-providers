from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from llm_providers.provider import LLMProvider

GOOSEAI_MODELS = {"fairseq-13b", "gpt-neo-20b", "gpt-j-6b"}

GOOSEAI_PARAMS = {
    "model": ("model", "gpt-neo-20b"),
    "max_tokens": ("max_tokens", 16),
    "min_tokens": ("min_tokens", 1),
    "temperature": ("temperature", 1.0),
    "n": ("n", 1),
    "top_p": ("top_p", 1.0),
    "top_k": ("top_k", 0),
    "tfs": ("tfs", 1.0),
    "top_a": ("top_a", 1.0),
    "typical_p": ("typical_p", 1.0),
    "stop": ("stop", None),
    "logit_bias": ("logit_bias", None),
    "logprobs": ("logprobs", None),
    "presence_penalty": ("presence_penalty", 0.0),
    "frequency_penalty": ("frequency_penalty", 0.0),
    "repetition_penalty": ("repetition_penalty", 1.0),
    "repetition_penalty_slope": ("repetition_penalty_slope", 0),
    "repetition_penalty_range": ("repetition_penalty_range", 0),
}


class GooseAIProvider(LLMProvider):
    def connect(
        self,
        connection_str: str,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the OpenAI server.

        connection_str is passed as default OPENAI_API_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.api_key = connection_str
        self.base_url = "https://api.goose.ai/v1"
        for key in GOOSEAI_PARAMS:
            setattr(self, key, client_args.pop(key, GOOSEAI_PARAMS[key][1]))
        if getattr(self, "model") not in GOOSEAI_MODELS:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {GOOSEAI_MODELS}.",
            )

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"model_name": "openai", "model": getattr(self, "model")}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(GOOSEAI_PARAMS.keys())

    # TODO: Parse response
    def prepare_request(
        self,
        query: str,
        request_args: Dict[str, Any] = {},
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        api_url = self.base_url + "/engines/" + getattr(self, "model") + "/completions"
        request_params = {"prompt": query}
        for key in GOOSEAI_PARAMS:
            request_params[GOOSEAI_PARAMS[key][0]] = request_args.pop(
                key,
                getattr(self, key),
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Cohere-Version": "2021-11-08",
        }
        return api_url, request_params, headers
