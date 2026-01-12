# Step1: Install & Import dependencies
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import create_react_agent
from arxiv_tool import arxiv_search
from read_pdf import read_pdf
from write_pdf import render_latex_pdf
import os
from dotenv import load_dotenv

load_dotenv()

# Step2: Setup LLM and tools
tools = [arxiv_search, read_pdf, render_latex_pdf]
llm= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)

# Step3: Run the agent with an initial prompt

INITIAL_PROMPT = """
You are an expert researcher in the fields of physics, mathematics,
computer science, quantitative biology, quantitative finance, statistics,
electrical engineering and systems science, and economics.

You are going to analyze recent research papers in one of these fields in
order to identify promising new research directions and then write a new
research paper. For research information or getting papers, ALWAYS use arxiv.org.
You will use the tools provided to search for papers, read them, and write a new
paper based on the ideas you find.

To start with, have a conversation with me in order to figure out what topic
to research. Then tell me about some recently published papers with that topic.
Once I've decided which paper I'm interested in, go ahead and read it in order
to understand the research that was done and the outcomes.

Pay particular attention to the ideas for future research and think carefully
about them, then come up with a few ideas. Let me know what they are and I'll
decide what one you should write a paper about.

Finally, I'll ask you to go ahead and write the paper. Make sure that you
include mathematical equations in the paper. Once it's complete, you should
render it as a LaTeX PDF. When you give papers references, always attatch the pdf links to the paper"""

# Step4: Create the ReAct agent graph
graph = create_react_agent(model, tools=tools)


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        print(f"Message received: {message.content[:200]}...")
        message.pretty_print()

while True:
    user_input = input("User: ")
    if user_input:
        input_data = {
            "messages": [
                {"role": "system", "content": INITIAL_PROMPT},
                {"role": "user", "content": user_input}
            ]
        }
        print_stream(graph.stream(input_data, stream_mode="values"))
