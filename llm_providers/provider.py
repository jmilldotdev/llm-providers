from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List

import aiohttp

from llm_providers.models import Completion
from llm_providers.models import PreparedRequest


class LLMProvider(ABC):
    def __init__(self, connection_str, client_args: Dict[str, Any] = {}):
        """
        Initialize client.

        kwargs are passed to client as default parameters.

        For clients like OpenAI that do not require a connection,
        the connection_str can be None.

        Args:
            connection_str: connection string for client.
            client_args: client arguments.
        """
        self.connect(connection_str, client_args)

    @abstractmethod
    def connect(self, connection_str: str, client_args: Dict[str, Any]) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare_request(
        self,
        prompt: str,
        request_args: Dict[str, Any] = {},
    ) -> PreparedRequest:
        """
        Prepare request.

        Args:
            prompt: prompt string.
            request_args: request arguments.

        Returns:
            api_url: url to send request to.

        """
        raise NotImplementedError()

    @abstractmethod
    def parse_completion(
        self,
        response: Dict[str, Any],
    ) -> str:
        """
        Formats the response object to return a completion string.

        Args:
            response: response object.

        Returns:
            Completion text.

        """
        raise NotImplementedError()

    async def complete(
        self,
        prompt: str,
        request_args: Dict[str, Any] = {},
    ) -> Completion:
        async with aiohttp.ClientSession() as session:
            prepared_request = self.prepare_request(prompt, request_args)
            async with session.post(
                prepared_request.api_url,
                json=prepared_request.params,
                headers=prepared_request.headers,
            ) as resp:
                response = await resp.json()
                completion_text = self.parse_completion(response)
                return Completion(
                    prompt=prompt,
                    completion_text=completion_text,
                    response=response,
                    params=prepared_request.params,
                )
