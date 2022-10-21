import asyncio
import os

from dotenv import load_dotenv

from llm_providers.providers.gooseai import GooseAIProvider


async def main():
    load_dotenv()
    provider = GooseAIProvider(
        connection_str=os.environ.get("GOOSEAI_API_KEY"),
    )

    completion = await provider.complete(query="hey i'm a dog who")
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
