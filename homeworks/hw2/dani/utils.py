from litellm import completion

# call llm
def call_llm(messages: list[dict], model: str = "gpt-4o-mini"):
    response = completion(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content
