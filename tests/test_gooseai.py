import asyncio
import os

from dotenv import load_dotenv

from llm_providers.gooseai import GooseAIProvider


async def main():
    load_dotenv()
    provider = GooseAIProvider(
        connection_str=os.environ.get("GOOSEAI_API_KEY"),
    )

    completion = await provider.complete(prompt="hey i'm a robot who")
    assert completion.prompt == "hey i'm a robot who"
    assert type(completion.completion_text) == str


if __name__ == "__main__":
    asyncio.run(main())
