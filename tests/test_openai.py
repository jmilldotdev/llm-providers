import asyncio
import os

from dotenv import load_dotenv

from llm_providers.openai import OpenAIProvider


async def main():
    load_dotenv()
    provider = OpenAIProvider(
        connection_str=os.environ.get("OPENAI_API_KEY"),
    )

    completion = await provider.complete(prompt="hey i'm a robot who")
    assert completion.prompt == "hey i'm a robot who"
    assert type(completion.completion_text) == str


if __name__ == "__main__":
    asyncio.run(main())
