from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from llm_providers.provider import LLMProvider


AI21_ENGINES = {
    "j1-jumbo",
    "j1-grande",
    "j1-large",
}

# User param -> (client param, default value)
AI21_PARAMS = {
    "engine": ("engine", "j1-large"),
    "temperature": ("temperature", 1.0),
    "max_tokens": ("maxTokens", 10),
    "top_k_return": ("topKReturn", 0),
    "n": ("numResults", 1),
    "top_p": ("topP", 1.0),
    "stop_sequences": ("stopSequences", []),
}


class AI21Provider(LLMProvider):
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
        self.base_url = "https://api.ai21.com/studio/v1"
        for key in AI21_PARAMS:
            setattr(self, key, client_args.pop(key, AI21_PARAMS[key][1]))
        if getattr(self, "engine") not in AI21_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {AI21_ENGINES}.",
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
        return list(AI21_PARAMS.keys())

    # TODO: Coerce to normal response
    def prepare_request(
        self,
        query: str,
        request_args: Dict[str, Any] = {},
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        api_url = self.base_url + "/" + getattr(self, "engine") + "/complete"
        request_params = {"prompt": query}
        for key in AI21_PARAMS:
            request_params[AI21_PARAMS[key][0]] = request_args.pop(
                key,
                getattr(self, key),
            )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return api_url, request_params, headers
