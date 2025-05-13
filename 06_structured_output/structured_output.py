# First we initialize the model we want to use.
from langchain_anthropic import ChatAnthropic
import getpass
import os

from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent

# setup environment ----------------------------------------------

def _set_env(var: str, file_name):
    if not os.environ.get(var):
        try:
            with open(file_name, "r") as file:
                os.environ[var] = file.read().strip()
        except FileNotFoundError:
            os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY", file_name="../anthropic_api_key.txt")

# define tools ----------------------------------------------------

model = ChatAnthropic(model="claude-3-haiku-20240307") # fast, cheap, less accurate


@tool
def get_weather(city: Literal["Berlin", "Paris"]):
    """Use this to get weather information."""
    if city == "Berlin":
        return "There is a snow storm in Berlin"
    elif city == "Paris":
        return "It's very hot and sunny in Paris"
    else:
        raise AssertionError("Unknown city")
tools = [get_weather]


# Define the structured output schema

class WeatherResponse(BaseModel):
    """Respond to the user in this format."""
    conditions: str = Field(description="My weather conditions")
    temperature: str = Field(description="Temperature in Celsius")
    wind: str = Field(description="wind conditions")
    humidity: str = Field(description="humidity in percent")
    city: str = Field(description="City for which the weather is reported")


# Define the graph
graph = create_react_agent(
    model,
    tools=tools,
    response_format=WeatherResponse,
)

# plot the graph as a nice png
with open("state_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


inputs = {"messages": [("user", "What's the weather in Berlin?")]}
response = graph.invoke(inputs)

print(">> Human readable response:")
for m in response["messages"]:
    print(m)


print(">> structured response:")
print(response["structured_response"])