import getpass
import os

from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition

# NEW
from typing_extensions import Literal
from langgraph.types import Command, interrupt
# we use the langchain capability to interrupt the graph now


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

# defines a tool that (allways) interrupt()s the flow 
# to requests additional human input
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


## a simple tool that searches for the weather
@tool
def weather_search(city: str):
    """Search for the weather"""
    print(f">> Weather tool is searching for: {city}")
    # return "Sunny"
    return f"Weather in {city} is sunny today!"

# another tool that searches for information online
tavi_tool = TavilySearchResults(max_results=1)


tools = [weather_search]

# -----------------------------------------------------------------------

# define a node that, per se, does nothing except
# interrupting the flow and collect input from a human
# In the end, it returns a Command object that tells the graph
# where to go next.
def human_review_node(state) -> Command[Literal["chatbot", "run_tool"]]:
    
    # get info on context that brought us here
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[-1]

    # this is the value we'll be providing via Command(resume=<human_review>)
    human_review = interrupt(
        {
            "question": "Is this correct?",
            # show the tool that calls for human review
            "tool_call": tool_call,
        }
    )

    review_action = human_review["action"]
    review_data = human_review.get("data")

    # if approved, call the tool
    if review_action == "continue":
        return Command(goto="run_tool")

    # update the AI message AND call tools
    elif review_action == "update":
        updated_message = {
            "role": "ai",
            "content": last_message.content,
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    # This the update provided by the human
                    "args": review_data,
                }
            ],
            # This is important - this needs to be the same as the message you replacing!
            # Otherwise, it will show up as a separate message
            "id": last_message.id,
        }
        return Command(goto="run_tool", update={"messages": [updated_message]})

    # provide feedback to LLM
    elif review_action == "feedback":
        # NOTE: we're adding feedback message as a ToolMessage
        # to preserve the correct order in the message history
        # (AI messages with tool calls need to be followed by tool call messages)
        tool_message = {
            "role": "tool",
            # This is our natural language feedback
            "content": review_data,
            "name": tool_call["name"],
            "tool_call_id": tool_call["id"],
        }
        return Command(goto="chatbot", update={"messages": [tool_message]})


def run_tool(state):

    new_messages = []
    tools = {"weather_search": weather_search}

    tool_calls = state["messages"][-1].tool_calls
    
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        new_messages.append(
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": result,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": new_messages}


# conditional edge to route to the human review node
# if the last message has tool calls. Otherwise, route to the end.
def route_after_llm(state) -> Literal[END, "human_review_node"]:
    if len(state["messages"][-1].tool_calls) == 0:
        return END
    else:
        return "human_review_node"


llm = ChatAnthropic(model="claude-3-haiku-20240307") # fast, cheap, less accurate
llm_with_tools = llm.bind_tools(tools)

# ----------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def stream_graph_updates(graph, config, user_input):

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

def main():

    builder = StateGraph(State)
    builder.add_node(chatbot)
    builder.add_node(run_tool)
    builder.add_node(human_review_node)
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", route_after_llm)
    builder.add_edge("run_tool", "chatbot")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)


    # plot the graph as a nice png
    with open("state_graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())


    config = {"configurable": {"thread_id": "1"}}

    # run graph as full chatbot - as before
    # while True:
    #     user_input = input("User: ")
    #     if user_input.lower() in ["quit", "exit", "q"]:
    #         print("Goodbye!")
    #         break
    #     else:
    #         stream_graph_updates(graph, config, user_input)


    # Example: Talk to bot without human review:
    print("\n--- Calling chatbot without need for interrupt -----------------------------")
    initial_input = {"messages": [{"role": "user", "content": "hi!"}]}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    for event in graph.stream(initial_input, thread, stream_mode="updates"):
        print(f"- {event}")

    # Example: Talk to bot with human review:
    print("\n--- Calling chatbot with need for interrupt -----------------------------")
    initial_input = {"messages": [{"role": "user", "content": "what's the weather in Paris?"}]}

    
    thread = {"configurable": {"thread_id": "2"}}
    graph_run = graph.stream(initial_input, thread, stream_mode="updates")

    print("\n>> Running the graph until the first interruption")
    for event in graph_run:
        print(f"- {event}")

    print("\n>> There are pending executions!")
    print(print(f"- {graph.get_state(thread).next}"))

    # simulate a human saying "yes, that's correct, continue"
    graph_run_follow_up = graph.stream( Command(resume={"action": "continue"}), thread, stream_mode="updates")
    
    # alternative: could also simulate a human saying "no, that's not correct, update"
    # graph_run_follow_up = graph.stream( Command(resume={"action": "update", "data": {"city": "Berlin"}}), thread, stream_mode="updates")
    
    print("\n>> Running the graph until the next interruption")
    for event in graph_run_follow_up:
        print(f"- {event}")
    
    
if __name__ == "__main__":
    main()