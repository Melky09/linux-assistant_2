from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

from utils import execute_command
from utils.speech_to_text import stt

from dotenv import load_dotenv
from prompt import prompt
import getpass
import os

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass(
        "Enter your Google AI API key: "
    )

# --------------------------------------------------
# LLM + TOOLS
# --------------------------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")

search = DuckDuckGoSearchResults(
    description="Search the internet for real-time information."
)

@tool
def gui_interaction(coordinates: str) -> str:
    """Performs a GUI action (currently not implemented)."""
    return "Command failed"

tools = [search, execute_command, gui_interaction]

# --------------------------------------------------
# AGENT
# --------------------------------------------------
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

graph = create_agent(
    llm,
    tools=tools,
    system_prompt=prompt,
    checkpointer=memory,
    debug=False,
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


def get_user_input():
    """
    Choose input mode:
    - Enter → text
    - v     → voice
    - q     → quit
    """
    mode = input("\n[Enter]=Text | v=Voice | q=Quit : ").strip().lower()

    if mode == "q":
        return None

    if mode == "v":
        print("🎙️ Listening...")
        text = stt()
        print(f"📝 You said: {text}")
        return text

    return input("You: ").strip()


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
if __name__ == "__main__":
    while True:
        user_text = get_user_input()

        if user_text is None:
            print("👋 Exiting assistant")
            break

        if not user_text:
            continue

        inputs = {"messages": [("user", user_text)]}
        print_stream(
            graph.stream(inputs, config=config, stream_mode="values")
        )