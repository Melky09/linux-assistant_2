from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from utils import stt, execute_command 
from dotenv import load_dotenv
from prompt import prompt
import getpass
import os

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

llm = ChatGroq(model="llama-3.3-70b-versatile") # Replace with your chosen model
search = DuckDuckGoSearchResults(description="""A web search tool for getting all kind of infromation on the internet. 
                                    You can use it get current events information and search for something you don't know.""")

@tool
def gui_interaction(coordinates: str) -> str:
    """Performs a certain gui action given the coordinates to a button on screen but is slow """
    return f"Comand failed"

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

tools = [search, execute_command, gui_interaction]

graph = create_agent(llm, tools=tools, system_prompt=prompt, checkpointer=memory, debug=False)

if __name__ == "__main__":
    while True:
        req = input("You: ")
        if(req == "a"): 
            req = stt()
            if(input(req + " ") == 'n'): continue
        if(req == "q"): break
        if(req == ""): continue
        inputs = {"messages": [("user", req)]}
        print_stream(graph.stream(inputs, config=config, stream_mode="values"))
