import getpass
import os

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_anthropic import ChatAnthropic

# NEW
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from langchain_core.messages import ToolMessage



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

# ----------------------------------------------------------------

# define a node that runs the tools requested in the last AImessage
# this code can be simplified by using the `ToolNode` class from langchain_core
# but we will define it manually here for demonstration purposes
class BasicToolNode:
    
    # gets inialized by the user with a list of tools
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    # gets called with a dictionary of inputs (= commands, messages)
    # and hands them over to the tools
    # returns a dictionary of outputs of the respective tools
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# ----------------------------------------------------------------

# chatbot node function takes the current State as input and 
# returns a dictionary containing an updated messages list 
# under the key "messages"
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# helper function to print graph updates (= chatbot responses)
# as they happen
def stream_graph_updates(graph, user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# Use in the conditional_edge to route to the ToolNode if the last message
# has tool calls. Otherwise, route to the end.
def route_tools(
    state: State,
):
    # if the current state is in a suitable "list of messages" format or
    # if the current state is not, but contains a list of messages
    # get the last message. 
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    # if the last message has tool calls, route to the tools node
    # else return to the END node
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    else:
        return END


def main():

    # build the graph
    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    
    tool_node = BasicToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_edge(START, "chatbot")

    # The `route_tools` function above returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", END: END},
    )

    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()

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