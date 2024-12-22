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
text = """National Center for Artificial Intelligence and Robotics (NCAIR)

1. General Information:
The official website is at https://ncair.nitda.gov.ng/.
Contact information includes the phone numbers: +2348178778499, +2348178778501, and email: ncair@nitda.gov.ng
The National Centre for Artificial Intelligence and Robotics (NCAIR) is one of NITDA’s special purpose vehicles created to promote research and development on emerging technologies and their practical application in areas of Nigerian national interest. The centre, a state-of-the-art facility, along with its modern digital fabrication laboratory (FabLab), was commissioned on November 13th, 2020. It is co-located in the same building complex with the Office for Nigerian Digital Innovation (ONDI), at No. 790 Cadastral Zone, Wuye District, Abuja.
NCAIR as a digital innovation and research facility is focused on Artificial Intelligence (AI), Robotics and Drones, Internet of Things (IoT), and other emerging technologies, aimed at transforming the Nigerian digital economy, in line with the National Digital Economy Policy and Strategy (NDEPS). NCAIR is also focused on creating a thriving ecosystem for innovation-driven entrepreneurship (IDE), job creation, and national development.

2. NCAIR Mission and Vision:
Mission: To promote research, development, and adoption of AI, robotics, and 4IR technologies for economic growth, improved quality of life, and global competitiveness.
Vision: To be a leading hub for Artificial Intelligence, Robotics, and other Fourth Industrial Revolution Technologies in Nigeria and Africa.

3. NCAIR Core Values:
Innovation: foster an environment for trailblazing ideas and solutions.
Collaboration: building partnerships and connections to amplify our impact.
Inclusiveness: commitment to an equitable society through technology.
Entrepreneurship: enabling technology commercialisation and enterprise

4. NCAIR Leadership:
The National Director of NCAIR is Dr. Olubunmi Ajala. He's a Data Scientist and Economist with high level experience providing expertise in National Strategy, policies, and programs relating to Artificial Intelligence and Data, Research and Digital Public Infrastructure. He leverages research \an data-driven insights to craft strategies for enhancing productivity using digital technologies.
He previously lectured and served as Learning Analytics Lead at the School of Economics, Finance and Accounting, Coventry University (UK) and also facilitated several Data Science training in Nigeria and Namibia. He is a member of the International AI Safety Team, working on the “State of the Science Report,” and was part of the Technological Innovation Group of the AU-EU Research and Innovation Advisory Group for “Mainstreaming Research and Innovation in Africa-European Cooperation”. 
Some of his other projects include:
Led the data team that designed the Composite Cultural Vibrancy Index for 12 African countries 
Led the team that built the Country and Product Opportunity Index for African trade.  
Developed a predictive model to detect HIV from TB (using AI), 
Built an Algorithm for identifying AI researchers of Nigerian descent
Built an algorithm for Anomaly Detection for Personal Banking Customers, and tracking Inflation via E-commerce websites. 
Dr. Ajala is the lead author of the article that used Twitter data to analyse public sentiments about the Africa Free Continental Free Trade Agreement (AfCFTA) and has extensive experience and competence in  handling structured and unstructured data and developing Machine Learning algorithms

5. NCAIR Key Areas of Focus:
Computer Vision
Robotics
Machine Learning
Data Science
Health AI
Natural Language Processing
Agriculture AI
Education AI
Finance AI
Trade AI
Unmanned Aerial Vehicles
IoT
Blockchain
Finance AI
Security AI

6. NCAIR current infrastructure:
The Centre is leveraging the existing national infrastructure of Galaxy Backbone. Our current Cloud Capacity enables us to commence pilot AI projects in our priority research areas.

Computing: 2,294vCPUs
Memory: 2,898GB (RAM)
Storage: 260TB (SSD)

7. NCAIR Research and Development:
AI in Healthcare
Robotics in Agriculture
Smart Cities and Automation
AI in Education

8. NCAIR Training Programs and Workshops:  
Python
Data Science
Embedded Systems
Product Design

10. WORK DONE SO FAR
Efforts on Artificial Intelligence & Robotics:
a. Fine Tuning Llama2 for specific Nigerian purposes: We are Fine-tuning LlaMa-2-70B with an initial focus on Nigerian trade by using a Nigerian trade dataset ( from the product level over the last 20 years) and the Social Connectedness Data (from Meta) to further train our model.
Our next step is to open source the next version of Llama 2 and make it available free of charge for research and commercial use. We will also be including model weights and starting code for the pretrained models and conversational fine-tuned versions.

b. Use of Machine Learning to identify top AI researchers of Nigerian descent.
Using machine learning to identify top AI researchers of Nigerian descent involves leveraging algorithms and data analysis techniques to sift through vast amounts of information to pinpoint individuals who have made significant contributions to the field of artificial intelligence while being of Nigerian heritage.
This process typically begins with gathering relevant data, such as academic publications, conference presentations, patents, and other indicators of research impact. Next, machine learning algorithms can be employed to analyze this data, identifying patterns and characteristics that are associated with top researchers in the field. These patterns might include citation counts, collaboration networks, publication venues, and the impact of their work on the broader scientific community.

c. Leveraging Meta Llama 3 LLM For Verifying And Validating Products in the Pharmaceutical Value Chain In Nigeria
The pharmaceutical value chain in Nigeria faces significant challenges such as drug counterfeiting, inefficient drug distribution, and the need for real-time data aggregation. To address these issues, the Design Lab (CcHUB) and the National Center for Artificial Intelligence and Robotics (NCAIR) propose leveraging Meta’s Llama 3 AI to create a digital platform for verifying and validating pharmaceutical products.
"""

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