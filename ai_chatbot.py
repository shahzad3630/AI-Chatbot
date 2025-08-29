import streamlit as st

from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from operator import itemgetter

# *********************************************************************************************

# set the title
st.set_page_config(page_title="Chatbot")
st.header("AI Chatbot")

# select the model
repo_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
model_file_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"


# *********************************************************************************************


# function to init chat model
@st.cache_resource
def create_chatbot():    
    model_path = hf_hub_download(repo_id=repo_id, filename=model_file_name, repo_type="model")

    model = LlamaCpp(model_path=model_path, temperature=0, max_tokens=512, 
                        top_p=1, verbose=False, streaming=True, stop=["Human:"] )


    system_prompt = "You are a AI bot who answers questions with human conversational style."

    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{human_input}"),
            ("ai", ""),
        ])


    buffer = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    def save_to_buffer(inputs_outputs):
        inputs = {"human": inputs_outputs["human"]}
        outputs = {"ai": inputs_outputs["ai"]}
        buffer.save_context(inputs, outputs)


    def get_response(response):
        return response["ai"]

    model_chain = {
            "human_input": RunnablePassthrough(),
            "chat_history": (
                RunnableLambda(buffer.load_memory_variables) |
                itemgetter("chat_history")
            )
        } | prompt | model


    bot_chain = RunnablePassthrough() | {
                "human": RunnablePassthrough(),
                "ai": model_chain
            } | {
                "save_memory": RunnableLambda(save_to_buffer),
                "ai": itemgetter("ai")
            } | RunnableLambda(get_response)

    return bot_chain


# *********************************************************************************************


# initialize the chatbot model
chatbot = create_chatbot()


if "messages" not in st.session_state:
    st.session_state.messages = [ {"role": "assistant", "content": "Hi, how can I help you?"} ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""


for message in st.session_state.messages:
    # if message["role"]=="system":
    #     continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# *********************************************************************************************


# ask for user input
user_prompt = st.chat_input("Your text message", key="user_input")
if user_prompt:

    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    response = chatbot.invoke(user_prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
