import getpass
import os

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_anthropic import ChatAnthropic


# setup environment ----------------------------------------------

# Load the API key from system or a file or prompt the user for it
def _set_env(var: str):
    if not os.environ.get(var):
        try:
            with open("../anthropic_api_key.txt", "r") as file:
                os.environ[var] = file.read().strip()
        except FileNotFoundError:
            os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY")


# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620") # slow, expensive, most accurate
llm = ChatAnthropic(model="claude-3-haiku-20240307") # fast, cheap, less accurate

# ----------------------------------------------------------------

#  A StateGraph object defines the structure of our chatbot 
# as a "state machine". We here define the schmea of the state.
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# chatbot node function takes the current State as input and 
# returns a dictionary containing an updated messages list 
# under the key "messages"
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# helper function to print graph updates (= chatbot responses)
# as they happen
def stream_graph_updates(graph, user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

def main():

    # build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile()

    # plot the graph as a nice png
    with open("state_graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    # run graph as full chatbot
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        else:
            stream_graph_updates(graph, user_input)

if __name__ == "__main__":
    main()