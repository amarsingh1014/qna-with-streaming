import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import asyncio

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGHAIN_PROJECT"] = os.getenv("LANGHAIN_PROJECT", "langchain-llm-app")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps answer user's questions."),
        ("human", "{input}"),
    ]
)

async def generate_response(question, api_key, llm, temperature, max_tokens):
    # Create the Groq LLM instance
    llm = ChatGroq(
        model=llm,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    response = ""
    async for event in chain.astream_events({"input": question}):
        print("DEBUG: Event received:", event)  # Debugging

        kind = event.get("event")  # Safely get the event type
        if kind == "on_chat_model_stream":
            # Extract the chunk content
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and hasattr(chunk, "content"):
                response += chunk.content  # Append the content
                yield response  # Stream the response incrementally

st.title("Q&A Chatbot With Open Source LLMs")

st.sidebar.title("Settings")

llm = st.sidebar.selectbox(
    "Select LLM",
    ["meta-llama/llama-4-scout-17b-16e-instruct", 
     "deepseek-r1-distill-llama-70b",
     "mistral-saba-24b",
     "gemma2-9b-it"],
    index=0,
)

temperature = st.sidebar.slider(min_value=0.0, max_value=2.0, value=0.5, label="Temperature")
max_tokens = st.sidebar.slider(min_value=0, max_value=500, value=100, label="Max Tokens")

st.write("## Ask a question")

question = st.text_input("Question", "What is the capital of France?")

if question:
    response_placeholder = st.empty()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def display_response():
        async for partial_response in generate_response(
            question,
            os.getenv("GROQ_API_KEY"),
            llm,
            temperature,
            max_tokens
        ):
            response_placeholder.text(partial_response)
    loop.run_until_complete(display_response())
    
else :
    st.write("Please enter a question to get a response.")
