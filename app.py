import streamlit as st
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import deque

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-model")
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token

# Load the Sales Shipment Data CSV
order_data = pd.read_csv("./Sales_Shipment_Data.csv")
order_data['Order Number'] = order_data['Order Number'].astype(str).str.zfill(5)

def generate_response(prompt, max_length=50):
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors='pt', padding=True)
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    input_ids = input_ids.to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    output_ids = model.generate(
        input_ids,
        max_length=max_length + input_ids.shape[1],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        top_k=60,
        no_repeat_ngram_size=2
    )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Find the last sentence-ending punctuation and preserve as many sentences as possible
    response = truncate_at_sentence_end(response)

    return response.strip()

def truncate_at_sentence_end(response):
    sentence_endings = list(re.finditer(r'[.!?]', response))
    if sentence_endings:
        last_ending = sentence_endings[-1]
        return response[:last_ending.end()]
    else:
        return response

def extract_order_id(user_input):
    match = re.search(r'\b\d{5}\b', user_input)
    if match:
        return match.group(0)
    return None

def extract_tracking_number(user_input):
    match = re.search(r'\bTRK\d{9}\b', user_input)
    if match:
        return match.group(0)
    return None

def post_process_response(response, 
                          order_number=None, 
                          order_status=None, 
                          order_tracking=None, 
                          eta=None,  
                          purchase_status=None, 
                          delivery_date=None, 
                          track_order=None, 
                          shipping_status=None, 
                          tracking_number=None,
                          customer_fname=None):
    
    if order_number:
        response = response.replace('<ORDER_NUMBER>', order_number)
    if order_status:
        response = response.replace('<ORDER_STATUS>', order_status)
    if order_tracking:
        response = response.replace('<ORDER_TRACKING>', order_tracking)
    if eta:
        response = response.replace('<ETA>', eta)
    if purchase_status:
        response = response.replace('<PURCHASE_STATUS>', purchase_status)
    if delivery_date:
        response = response.replace('<DELIVERY_DATE>', delivery_date)
    if track_order:
        response = response.replace('<TRACK_ORDER>', track_order)
    if shipping_status:
        response = response.replace('<SHIPPING_STATUS>', shipping_status)
    if tracking_number:
        response = response.replace('<TRACKING_NUMBER>', tracking_number)
    if customer_fname:
        response = response.replace('<CUSTOMER_FNAME>', customer_fname)

    response = re.sub(r'<[^>]+>', '', response)
    return response

def clean_generated_response(response):
    clean_response = []
    for sentence in response.split('.'):
        cleaned_sentence = re.sub(r'<[^>]*>', '', sentence)
        cleaned_sentence = cleaned_sentence.replace('_', ' ')
        clean_response.append(cleaned_sentence.strip())
    
    return '. '.join(clean_response).strip()

def get_order_details(order_number=None, tracking_number=None):
    if order_number:
        result = order_data[order_data['Order Number'] == order_number]
    elif tracking_number:
        result = order_data[order_data['Tracking Number'] == tracking_number]
    else:
        return None, None
    
    if not result.empty:
        order_status = result.iloc[0]['Order Status']
        eta = result.iloc[0]['Estimated Delivery Date']
        shipping_status = result.iloc[0]['Delivery Status']
        customer_fname = result.iloc[0]['Customer Fname']
        order_number = result.iloc[0]['Order Number']
        
        return order_status, eta, shipping_status, customer_fname, order_number
    return None, None, None, None, None

# Streamlit app

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=6)

st.title("Customer Service Chatbot Application - Logistics Tracking Function")

st.write("Assistant: Hello! Welcome to the Order Tracking Assistant.How can I help you today?")

# Display conversation history
for message in st.session_state.conversation_history:
    st.write(message)

# Text input for the user
user_input = st.text_input("You: ")

if st.button("Send"):
    if user_input.lower() in ['exit', 'quit', 'bye', 'end']:
        st.write("Assistant: Thank you for using the Order Tracking Assistant. Have a great day!")
    elif user_input.lower() in ['thank', 'thanks', 'thanks.']:
        st.write("Assistant: My pleasure! Let me know if you need further help.")
    else:
        #conversation_history.append(f"Customer: {user_input}")
        st.session_state.conversation_history.append(f"You: {user_input}")
        
        order_number = extract_order_id(user_input)
        tracking_number = extract_tracking_number(user_input)
        
        if not order_number and not tracking_number:
            assistant_prompt = "\n".join(st.session_state.conversation_history) + "\n" + "Order Tracking Assistant: Provide a short and direct answer. Ask the user to provide both their order number and tracking number if they want to track an order."
            response = generate_response(assistant_prompt)
        else:
            order_status, eta, shipping_status, customer_fname, order_number = get_order_details(order_number, tracking_number)
            
            if not order_status or not eta:
                response = "I'm sorry, I couldn't find any details for the provided order number or tracking number. Please check and try again. Feel free to ask if you need further help or have any other questions."
            else:
                response_template = "Hi,<CUSTOMER_FNAME>.Your order number <ORDER_NUMBER> is currently <ORDER_STATUS>. The estimated delivery date is <ETA>. Is there anything else I can assist you with today? Feel free to ask if you need further help or have any other questions."
                response = post_process_response(
                    response_template, 
                    order_number=order_number, 
                    order_status=order_status, 
                    eta=eta,
                    customer_fname=customer_fname
                )
                response = clean_generated_response(response)
        
        st.session_state.conversation_history.append(f"Order Tracking Assistant: {response}")
        st.write(f"Order Tracking Assistant: {response}")
