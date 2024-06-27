from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

import os
from dotenv import load_dotenv
import json

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# Creating a tool named 'search'
search = TavilySearchResults()
tools = [search]

from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_KEY)

updated_prompt = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer.
Thought: you should always think about what may be the answers and their topic
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer which to be ended with exclamation !
Final Answer: The answer must not be in string format. The answers must separate with a exclamation!  .  Answer of the original question must be in a sentence. If there are multiple answers then give the output response
as a loopable python list form without numbers.No need to write 1. answer like this. The final output must be in python list where it shows list of answers.


Question: {input}
Thought:{agent_scratchpad}
"""

updated_prompt2 = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer.
Thought: you should always think about what may be the answers and their topic
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer 
Final Answer: The answer now must give proper explanation to the question. It can be multiple lines upto 4 lines at maximum . You can add the answers in bullet too.

Question: {input}
Thought:{agent_scratchpad}
"""



from langchain.prompts import PromptTemplate
new_prompt = PromptTemplate.from_template(updated_prompt)
new_prompt2 = PromptTemplate.from_template(updated_prompt2)

agent = create_react_agent(llm=llm, tools=tools, prompt=new_prompt)
agent2 = create_react_agent(llm=llm, tools=tools, prompt=new_prompt2)


agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
agent_executer2 = AgentExecutor(agent=agent2, tools=tools, verbose=True, handle_parsing_errors=True)

# Streamlit UI components
st.title("Interactive Chat with AI")
user_input = st.text_input("Enter your query here:")

if st.button('Search'):
    if user_input:
        # Step 1: Invoke the agent with user input
        result = agent_executer.invoke({"input": user_input})
        response_output = result['output']
        
        # Debug: Print the response output to the Streamlit console
        st.write("Response Output:", response_output)
        
        def convert_to_list(response):
            if isinstance(response, list):
                return response
            elif isinstance(response, str):
                # Example: Split string by periods and strip whitespace
                return [item.strip() for item in response.split('! ') if item]
            else:
                # Convert to string and split by periods as a fallback
                response_str = str(response)
                return [item.strip() for item in response_str.split('! ') if item]

        facts_list = convert_to_list(response_output)

        # Step 3: Display each fact
        for fact in facts_list:
            st.write(fact)

        # Store the list of facts in session state for later use
        st.session_state.facts_list = facts_list

if 'facts_list' in st.session_state:
    # Allow the user to select a fact from the previous search results
            selected_fact = st.selectbox("Select a fact to search again:", st.session_state.facts_list)

            if st.button('Search Again'):
                 if selected_fact:
            # Step 1: Invoke the agent with the selected fact
                    result = agent_executer2.invoke({"input": selected_fact})
                    response_output2 = result['output']
            
            # Debug: Print the response output to the Streamlit console
                    st.write("Response Output:", response_output2)       



