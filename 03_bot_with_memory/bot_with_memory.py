import getpass
import os

from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# NEW
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
# we use the langchain built in ToolNode now


# setup environment ----------------------------------------------

# Load the API key from system or a file or prompt the user for it
def _set_env(var: str, file_name):
    if not os.environ.get(var):
        try:
            with open(file_name, "r") as file:
                os.environ[var] = file.read().strip()
        except FileNotFoundError:
            os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY", file_name="../anthropic_api_key.txt")
_set_env("TAVILY_API_KEY", file_name="../tavily_api_key.txt")


# define tools ----------------------------------------------------

# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620") # slow, expensive, most accurate
llm = ChatAnthropic(model="claude-3-haiku-20240307") # fast, cheap, less accurate

tool = TavilySearchResults(max_results=1)
tools = [tool]
tool.invoke("How is the weather in Paris?")

# bind the tools to the language model - this allows the language model to use the quittools
# but it will not yet use them unless they are explicitly called in the graph
# we will have to add the tools to a new node
llm_with_tools = llm.bind_tools(tools)

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
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# helper function to print graph updates (= chatbot responses)
# as they happen
# This function is called any time e.g. the user inputs something.
# The events in the method contain everything that happened in the graph
def stream_graph_updates(graph, config, user_input):

    # NEW: add a config as second positional argument to stream()
    # the thread_id is used to store the state of the graph
    # in the memory checkpointer
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

def main():

    # build the graph
    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("tools", "chatbot")
    
    # NEW: add memory to the graph
    # create a MemorySaver checkpointer to save the state of the graph
    # this will work in-memory for now, but can be extended to save to disk
    # or a database (see SqliteSaver or PostgresSaver)
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)


    # plot the graph as a nice png
    with open("state_graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())


    config = {"configurable": {"thread_id": "1"}}

    # run graph as full chatbot - as before
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        else:
            stream_graph_updates(graph, config, user_input)

    print("\n\n Snapshot of the current graph state:")
    snapshot = graph.get_state(config)
    print(snapshot)


if __name__ == "__main__":
    main()