import asyncio
import os

from dotenv import load_dotenv

from llm_providers.openai import OpenAIProvider


async def main():
    load_dotenv()
    provider = OpenAIProvider(
        connection_str=os.environ.get("OPENAI_API_KEY"),
    )

    completion = await provider.complete(query="hey i'm a dog who")
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
