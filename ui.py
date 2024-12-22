import streamlit as st
from streamlit_chat import message as st_message

import spacy
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import pandas as pd

import os
import time


# Load data
########################################
data_path = '/data/ncair-data.txt'

with open(data_path, 'r') as file:
    text = file.read()


## Chunk data
########################################
num_parts = 10

full_stops_indices = [i for i, char in enumerate(text) if char == '.']
full_stops_per_part = len(full_stops_indices) // num_parts
split_indices = [full_stops_indices[(idx+1)*full_stops_per_part] for idx in range(num_parts-1)]

parts = [text[i:j] for i, j in zip([0] + split_indices, split_indices + [None])]


# Generate response
########################################
def answer_question(context, question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    inputs = tokenizer.encode_plus(
        question, context,
        return_tensors='pt',
        max_length=512,
        truncation=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = torch.softmax(outputs.start_logits, dim=1).cpu().numpy()[0]
    end_scores = torch.softmax(outputs.end_logits, dim=1).cpu().numpy()[0]

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    # Get the tokens as a list of strings
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])

    answer_tokens = all_tokens[start_index:end_index+1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    confidence = (start_scores[start_index] + end_scores[end_index]) / 2

    return answer, confidence


def reply(prompt, parts):
    answers_dict = {}
    for i, part in enumerate(parts):
        answer, confidence = answer_question(part, prompt)
        answers_dict[f"Chunk {i+1}"] = {answer :confidence}

        max_score = max(answers_dict.values(), key=lambda x: list(x.values())[0])
        max_value = list(max_score.keys())[0]
    return max_value


def botResponse(prompt):
    response = reply(prompt, parts=parts)
    return response


# Streamlit codes
########################################
st.title("NCAIRBot")




if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with unique keys for each message
for i, message in enumerate(st.session_state.messages):
    st_message(message["content"], is_user=(message["role"] == "user"), key=f"message_{i}")


# Handle user input
if prompt := st.chat_input():
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message with a unique key and custom avatar
    st_message(
        prompt, 
        is_user=True, 
        key=f"user_{len(st.session_state.messages)}"
    )

    # Generate bot response
    response = botResponse(prompt)

    # Display bot response once with a unique key and custom avatar
    st_message(response, is_user=False, key=f"assistant_{len(st.session_state.messages)}")

    # Append bot response
    st.session_state.messages.append({"role": "assistant", "content": response})