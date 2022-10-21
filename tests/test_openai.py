import asyncio
import os

from llm_providers.providers.openai import OpenAIProvider


async def main():
    provider = OpenAIProvider(
        connection_str=os.environ.get("OPENAI_API_KEY"),
    )

    inputs = provider.get_model_inputs()
    print(inputs)

    completion, request_args = await provider.complete(query="hey i'm a dog who")
    print(completion)
    print(request_args)


if __name__ == "__main__":
    asyncio.run(main())
