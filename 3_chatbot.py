# import basics
import os

# import streamlit
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from langchain import hub
# import langchain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from discover_ollama import discover_ollama

# load environment variables
load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url=discover_ollama()
)

###############################   INITIALIZE CHROMA VECTOR STORE   #############################################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)


###############################   INITIALIZE CHAT MODEL   #######################################################################################################

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0,
    base_url=discover_ollama()
)


# pulling prompt from hub
prompt = PromptTemplate.from_template("""
You are a helpful assistant. You will be provided with a query and a chat history.
Your task is to retrieve relevant information from the vector store and provide a response.
For this you use the tool 'retrieve' to get the relevant information.

The query is as follows:
{input}

The chat history is as follows:
{chat_history}

Please provide a concise and informative response based on the retrieved information.
If you don't know the answer, say "I don't know" (and don't provide a source).

You can use the scratchpad to store any intermediate results or notes.
The scratchpad is as follows:
{agent_scratchpad}

For every piece of information you provide, also provide the source.

Return text as follows:

<Answer to the question>
Source: source_url
""")


# creating the retriever tool
@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)

    serialized = ""

    for doc in retrieved_docs:
        serialized += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"

    return serialized


# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("How are you?")


# Count tokens in chat history context
def count_tokens(messages, model_name=None):
    # Use tiktoken to count tokens, fallback to simple count if not available
    try:
        enc = tiktoken.encoding_for_model(
            model_name or os.getenv("CHAT_MODEL", "gpt-3.5-turbo"))
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    text = "\n".join([m.content for m in messages])
    return len(enc.encode(text))


# did the user submit a prompt?
if user_question:
    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    context_token_count = count_tokens(st.session_state.messages)

    # invoking the agent
    result = agent_executor.invoke(
        {"input": user_question, "chat_history": st.session_state.messages})
    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(
            f"{ai_message}\n\n---\n_Context token count: {context_token_count}_")
        st.session_state.messages.append(AIMessage(ai_message))
