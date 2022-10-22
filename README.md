# llm-providers

A common, minimal-dependency, async-first interface for interacting with Large Language Model provider APIs.

## Getting Started

`pip install llm-providers`

```python
from llm_providers.openai import OpenAIProvider

async def main():
  provider = OpenAIProvider(
      connection_str=os.environ.get("OPENAI_API_KEY"),
  )

  completion = await provider.complete(prompt="hey i'm a robot who")
  assert completion.prompt == "hey i'm a robot who"
  assert type(completion.completion_text) == str

if __name__ == "__main__":
    asyncio.run(main())
```

## License

This project is licensed under the Apache License

## Acknowledgments

Much code borrowed from https://github.com/HazyResearch/manifest
