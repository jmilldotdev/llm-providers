import asyncio
import os

from dotenv import load_dotenv

from llm_providers.ai21 import AI21Provider


async def main():
    load_dotenv()
    provider = AI21Provider(
        connection_str=os.environ.get("AI21_API_KEY"),
    )

    completion = await provider.complete(query="hey i'm a dog who")
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
