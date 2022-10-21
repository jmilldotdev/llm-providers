from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import aiohttp


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
        query: str,
        request_args: Dict[str, Any] = {},
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError()

    async def complete(
        self,
        query: str,
        request_args: Dict[str, Any] = {},
    ) -> Tuple[Callable[[], Dict], Dict]:
        async with aiohttp.ClientSession() as session:
            api_url, request_params, headers = self.prepare_request(query, request_args)
            async with session.post(
                api_url,
                json=request_params,
                headers=headers,
            ) as resp:
                response = await resp.json()
                return response, request_params
