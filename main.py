import ollama
import os
import pandas as pd
import numpy as np
import time        
import streamlit as st
# response = ollama.chat(model='llama2', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])


#ollama.pull('nomic-embed-text')

df = pd.read_csv('data/Securityview_all.csv')

#df = df.loc[1:30,:]

# for index, row in df.iterrows():
#     print(row['Question'])

# embedding = ollama.embeddings(model='nomic-embed-text', prompt = 'How want to embed')

if not os.path.exists('embeddings.npy'):
    def get_embedding(question, answer):
        text = f"Question: {question} Answer: {answer}"
        return ollama.embeddings(model='nomic-embed-text', prompt = text)['embedding']

    embeddings = []
    for index, row in df.iterrows():
        question = row['Question']
        answer = row['Answer']
        embedding = get_embedding(question, answer)
        embeddings.append(embedding)

    # Convert list of embeddings to numpy array
    embeddings_array = np.array(embeddings)
    np.save('embeddings.npy', embeddings_array)
else:
    # Load embeddings from file
    embeddings_array = np.load('embeddings.npy')

def get_frm_vectordb(question,doc_size):
   text = f"Question: {question}"
   query =  ollama.embeddings(model='nomic-embed-text', prompt = text)['embedding']
   query = np.array(query).T
   final_embedd_result = df.iloc[np.argsort(np.dot(embeddings_array, query))[::-1][0:doc_size]]
   format_data = "\n\n".join([f"{row['Question']}:: {row['Answer']}" for index, row in final_embedd_result.iterrows()])
   return format_data

# print(get_frm_vectordb(question='How to add a controller',doc_size= 2))

def response(user_query):
    content = get_frm_vectordb(question=user_query,doc_size=2)
    response = ollama.chat(model='llama2', messages=[
    {"role": "system", "content": 'You are a helpful assistant'},
    {"role": "assistant", "content": content},
    {"role": "user", "content": user_query}
    ])
    output = response['message']['content']
    for word in output.split(" "):
            yield word + " "
            time.sleep(0.03)


## Streamlit-App

# from hugchat import hugchat
# from hugchat.login import Login

# App title
#st.set_page_config(page_title="🤗💬 HugChat")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
# Streamlit
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = response(prompt)
            st.write_stream(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)