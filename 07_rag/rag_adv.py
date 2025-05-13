
import getpass
import os

from langchain.chat_models import init_chat_model

# embedding
from langchain_core.embeddings import DeterministicFakeEmbedding
# from langchain_ollama import OllamaEmbeddings

# vector store
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_chroma import Chroma

# actual graph
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Annotated
from typing import Literal


# setup environment ----------------------------------------------

def _set_env(var: str, file_name):
    if not os.environ.get(var):
        try:
            with open(file_name, "r") as file:
                os.environ[var] = file.read().strip()
        except FileNotFoundError:
            os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY", file_name="../anthropic_api_key.txt")
_set_env("LANGSMITH_API_KEY", file_name="../langsmith_api_key.txt")


# define tools ----------------------------------------------------

llm = init_chat_model("claude-3-haiku-20240307", model_provider="anthropic")

# going for a cheap demo here, see https://python.langchain.com/docs/tutorials/rag/#langsmith 
# for more information
embeddings = DeterministicFakeEmbedding(size=4096)
# embeddings = OllamaEmbeddings(model="llama3")

# Define the vector store. Again, going for cheap option.
vector_store = InMemoryVectorStore(embeddings)
# vector_store = Chroma(embedding_function=embeddings)

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Define the graph ------------------------------------------------

# Load and chunk contents of some blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# annotate the sections of the document
# this is to demonstrate that we can annotate the splits with arbitrary metadata
total_documents = len(all_splits)
third = total_documents // 3
for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"


# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    query: Search                    # NEW
    context: List[Document]
    answer: str

# NEW: analyze and improve the query
def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


# NEW retrieve function
def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()



# plot the graph as a nice png
with open("state_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")
