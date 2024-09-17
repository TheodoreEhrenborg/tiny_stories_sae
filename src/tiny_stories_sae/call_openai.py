#!/usr/bin/env python3
from dotenv import load_dotenv
from pydantic import BaseModel


class EventDescription(BaseModel):
    location: str
    notable_fact: str


load_dotenv()
from openai import OpenAI

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ],
    response_format=EventDescription,
)

print(response)
