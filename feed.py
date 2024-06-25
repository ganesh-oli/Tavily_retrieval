# app.py
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

import os
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

#creating a tool named 'search'
search = TavilySearchResults()

tools = [search]

from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_KEY)

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

updated_prompt = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Final answer must include the answer and in next line include the URL of the answer.

Question: {input}
Thought:{agent_scratchpad}
"""

from langchain.prompts import PromptTemplate
new_prompt = PromptTemplate.from_template(updated_prompt)


agent = create_react_agent(llm=llm, tools=tools, prompt=new_prompt)

agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI components
st.title("Interactive Chat with AI")
user_input = st.text_input("Enter your query here:")

# Respond to user input
if user_input:
    # Invoke the agent with user input
    result = agent_executer.invoke({"input": user_input})
    
    # Display the response
    if result:
        st.subheader("Response:")
        st.write(result)
    else:
        st.write("No response generated.")


