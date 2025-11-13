"""
hierarchical_summary.py
Hierarchical summarization example using LangGraph + LangChain.
"""

import time
from typing import List, TypedDict, Optional
import os
from langgraph.graph import StateGraph
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ---- CONFIG ----
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")   # change if needed
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_CHUNK_SIZE = 2200            # characters per chunk (tune as needed)
CHUNK_OVERLAP = 200
MAP_SUMMARY_MAX_TOKENS = 300     # maximum tokens we allow for each chunk summary
REDUCE_ROUND_MIN = 2             # minimum number of summaries before stopping reduce
FINAL_SUMMARY_MAX_TOKENS = 1000

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your environment.")

# ---- State schema for the graph ----
class MyState(TypedDict, total=False):
    # inputs
    file_path: str
    user_question: Optional[str]
    # intermediate
    raw_documents: Optional[List[Document]]
    chunks: Optional[List[Document]]
    map_summaries: Optional[List[str]]
    # outputs
    final_summary: Optional[str]

# ---- Utilities ----
def load_document(path: str) -> List[Document]:
    """
    Load a document from file. Supports plain txt and pdf (via PyPDFLoader).
    Returns list of langchain Document objects.
    """
    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs = loader.load()
        return docs
    else:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into chunks using a recursive character splitter.
    We return a list of Documents (each with page_content and metadata).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

# ---- LLM helper: summarize a single chunk ----
def summarize_chunk(llm: ChatOpenAI, chunk_text: str, chunk_index: int) -> str:
    """
    Summarize a single chunk. Keep the prompt concise and instruct the model
    to return a short factual summary in 2-15 sentences.
    """
    prompt = PromptTemplate.from_template(
        "You are a concise summarizer. Summarize the following text in 2-15 short sentences, focusing on facts and key points. "
        "Do not invent facts. Output only the summary.\n\n"
        "Chunk index: {index}\n\n"
        "Text:\n{chunk}\n\n"
    )
    messages = prompt.format_prompt(index=chunk_index, chunk=chunk_text).to_messages()
    # call the LLM (langchain will use OPENAI_API_KEY)
    resp = llm.invoke(messages)  # Use invoke instead of generate
    # Extract text from response safely
    summary = resp.content.strip()
    return summary

# ---- LLM helper: combine summaries (reduce step) ----
def reduce_summaries(llm: ChatOpenAI, summaries: List[str]) -> str:
    """
    Combine multiple summaries into a higher-level summary.
    We also instruct the model to keep it short.
    """
    joined = "\n\n".join(f"- {s}" for s in summaries)
    prompt = PromptTemplate.from_template(
        "You are a hierarchical summarizer. Given the following list of short summaries, produce a"
        " concise, higher-level summary (max ~200-300 words) that synthesizes the main points. "
        "Do not hallucinate new facts. Output only the summary.\n\n"
        "Summaries:\n{joined}\n\n"
    )
    messages = prompt.format_prompt(joined=joined).to_messages()
    resp = llm.invoke(messages)
    return resp.content.strip()

# ---- Graph nodes ----
def node_load(state: MyState) -> MyState:
    """Node: load file and store raw documents in state."""
    path = state["file_path"]
    docs = load_document(path)
    return {"raw_documents": docs}

def node_split(state: MyState) -> MyState:
    """Node: split raw documents into chunks and store in state."""
    docs = state["raw_documents"]
    chunks = split_documents(docs)
    return {"chunks": chunks}

def node_map_summarize(state: MyState) -> MyState:
    """
    Node: summarize each chunk individually (map step).
    We store a list of chunk summaries in state['map_summaries'].
    """
    chunks: List[Document] = state["chunks"]
    # Create an LLM instance; ChatOpenAI will read OPENAI_API_KEY from env
    llm = ChatOpenAI(model=OPENAI_MODEL, max_tokens=MAP_SUMMARY_MAX_TOKENS)
    summaries = []
    for i, doc in enumerate(chunks):
        txt = doc.page_content
        s = summarize_chunk(llm, txt, i)
        summaries.append(s)
    return {"map_summaries": summaries}

def node_reduce_loop(state: MyState) -> MyState:
    """
    Node: perform hierarchical reduce until the number of summaries is small enough.
    We repeatedly compress groups of summaries into higher-level summaries.
    """
    llm = ChatOpenAI(model=OPENAI_MODEL, max_tokens=FINAL_SUMMARY_MAX_TOKENS)
    summaries = state["map_summaries"]
    # If there are few summaries, just combine them
    while len(summaries) > REDUCE_ROUND_MIN:
        # Combine summaries in batches of N to produce fewer summaries
        batch_size = max(2, min(6, len(summaries)//2))  # heuristic
        new_summaries = []
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            combined = reduce_summaries(llm, batch)
            new_summaries.append(combined)
        # set new summaries and continue loop if still large
        summaries = new_summaries
    # final top-level summary: combine what's left into final_summary
    final = reduce_summaries(llm, summaries)
    return {"final_summary": final}

# ---- Build and run StateGraph ----
def build_and_run_graph(file_path: str) -> str:
    """
    Build the LangGraph StateGraph with the nodes defined above, run it,
    and return the final summary.
    """
    # Create initial state
    initial_state: MyState = {"file_path": file_path}

    # Build graph
    graph = StateGraph(MyState)
    graph.add_node("load", node_load)
    graph.add_node("split", node_split)
    graph.add_node("map_summarize", node_map_summarize)
    graph.add_node("reduce", node_reduce_loop)

    # Wire nodes in order
    graph.add_edge("load", "split")
    graph.add_edge("split", "map_summarize")
    graph.add_edge("map_summarize", "reduce")
    
    # Set entry and finish points
    graph.set_entry_point("load")
    graph.set_finish_point("reduce")

    # Compile and run graph
    compiled_graph = graph.compile()
    result = compiled_graph.invoke(initial_state)  # synchronous execution
    final_summary = result.get("final_summary", "")
    return final_summary

# ---- CLI interface for quick testing ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hierarchical summarization using LangGraph.")
    parser.add_argument("file", help="Path to text or PDF file to summarize.")
    args = parser.parse_args()

    start = time.perf_counter()
    print("Running hierarchical summarization...")
    summary = build_and_run_graph(args.file)
    print("\n---- FINAL SUMMARY ----\n")
    print(summary)
    end = time.perf_counter()
    print(f"\nTime taken: {end - start:.2f} seconds")
    print("\n-----------------------\n")