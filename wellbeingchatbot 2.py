import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import cohere
import textwrap
st.title("Wellbeing Chatbot")
st.write("This is an AI wellbeing officer.")
st.subheader(
    "Go to this link to submit your feedback: ")
st.link_button("Feedback Form", "https://forms.gle/xF5FqbDUdFDo5qWa7")
st.subheader("Step 1: Setup your OpenAI API Key")
# ask for a user text input
user_openai_api_key = st.text_input("Enter your OpenAI API Key")
st.write("You can get yours from here - https://beta.openai.com/account/api-keys")


def generate_output(user_prompt):
    if user_openai_api_key is not None:
        co = cohere.Client('gF6ZjOmR1xqSNBFvKYs9tqEdnJKM44tKDnmt1rqf')

        base_prompt = textwrap.dedent("""
        You are an AI-powered wellbeing officer designed to engage in supportive conversations with employees. Your goal is to promote a positive and empathetic interaction. Respond to employee queries or concerns with a focus on providing comfort, advice, and encouragement. Incorporate active listening skills and offer appropriate guidance. Your tone should be friendly, understanding, and uplifting. Remember, your primary objective is to enhance the emotional well-being of the employees you interact with.

        Question:""")
        response = co.generate(
            model='command',
            prompt=base_prompt + " " + user_prompt + "\nAnswer: ",
            max_tokens=200,
            temperature=0.9,
            k=0,
            p=0.7,
            frequency_penalty=0.1,
            presence_penalty=0,
            stop_sequences=["--"])

        ai_output = response.generations[0].text
        ai_output = ai_output.replace("\n\n--", "").replace("\n--", "").strip()
        return ai_output


form = st.form(key="user_settings")
with form:
    st.write("Hi! What's up?")
    # User input - Question
    user_input = st.text_input("Question", key="user_input")

    # Submit button to start generating answer
    generate_button = form.form_submit_button("Submit Question")
    num_input = 1
    if generate_button:
        if user_input == "":
            st.error("Question cannot be blank")
        else:
            my_bar = st.progress(0.05)
            st.subheader("Answer:")

            for i in range(num_input):
                st.markdown("""---""")
                ai_output = generate_output(user_input)
                st.write(ai_output)
                my_bar.progress((i+1)/num_input)

st.write('')
