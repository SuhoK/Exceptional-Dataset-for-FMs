import os
import openai
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import requests
import re
import ctypes
from ctypes import wintypes
import json
import pandas as pd
import numpy as np
from IPython.display import display
import csv


api_key = "Enter your API Key"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

exprompt1="Step1: Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence.\n Identify that the sentence starts with '바라는게' and ends with '안그래? \n\n Step2: Split into words and translate each word in english and identify any typos based on the context.: \n바라는게 (What I hope for) 무한정 (infinitely) 끝없이 (endlessly)내리는 (falling)게 (particle, indicating 'is')아닌게 (is not)엄마나 (Typo: misidentified word, Correct: 얼마나, Translation: how much)다행인지(fortunately)몰라 (I don't know)안그래? (isn't it?)\n\n Step3: Correct the typos by comparing each word with similar words and combine the corrected words to form a coherent sentence.:\n '엄마나' should be '얼마나', '알고 래?' should be '안그래?'\n\nStep4:Combine based on context: \n '바라는게 무한정 끝없이 내리는 게 아닌게 얼마나 다행인지 몰라 안그래?' There is no weird word to use. \n\n Step5: Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one.\n Infer the context: The sentence talks about how fortunate it is that something is not happening endlessly. \n Correct any misrecognized words: '얼 마나' should be '얼마나', '알고 래?' should be '안그래?' \n\n Step6: Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense.\nCheck grammar and spelling: Ensure '바라는게 무한정 끝없이 내리는 게 아닌게 얼마나 다행인지 몰라 안그래?' is grammatically correct and makes sense. \nEnsure the sentence flows naturally and the meaning is clear."

exprompt2="Step1: Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence. \n Identify that the sentence starts with '해묵은' and ends with '오후 햇살이'. \n\n Step2: Split into words and translate each word in english and identify any typos based on the context.: 해묵은 (aged)나이를 (age)잊은 (forgotten)순간에 (in the moment)따라오는 (following)오후 (afternoon)햇살이 (the sunlight)There is no weird word to use. \n Step3: Correct the typos by comparing each word with similar words and combine the corrected words to form a coherent sentence.:\n There was no weird word \n\nStep4: Combine based on context: '해묵은 나이를 잊은 순간에 따라오는 오후 햇살이' \n\n Step5: Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one. \nInfer the context: The sentence talks about the nostalgic feeling of afternoon sunlight in a moment where old age is forgotten. \n Combine based on context: '해묵은 나이를 잊은 순간에 따라오는 오후 햇살이' \n Correct any misrecognized words: Ensure '해묵은' is correctly recognized and '오후 햇살이' is interpreted correctly \n\n Step6: Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense. Check grammar and spelling: \nEnsure '해묵은 나이를 잊은 순간에 따라오는 오후 햇살이' is grammatically correct and makes sense. \nEnsure the sentence flows naturally and the meaning is clear."

def gpt_api_cot_few(image_path_pr):
    image_path_ex = r"\example1,jpg"
    image_path_ex2= r"\example2.jpg"
    
    base64_image_ex = encode_image(image_path_ex)
    base64_image_ex2 = encode_image(image_path_ex2)
    base64_image_pr = encode_image(image_path_pr)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


    message = [
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text":"Below are examples of OCR task. \n I'll show image first and explain step-by-step how to extract text from the image."
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "\n\nFirst Example:"
            }]
        },
       {
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_ex}"
                }
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": exprompt1
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "\n\nSecond Example:"
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_ex2}"
                }
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": exprompt2
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Now, please perform an OCR task on the following image like the example. The image is Korean calligraphy with an illustration. \nTranscribe the letters in the picture with a step-by-step explanation of your reasoning. \n But Don't describe yout steps. Just answer the result of the OCR without translating."

        }]
        },
        {
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_pr}"
                }
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Let's think step by step. Transcribe the image please."

        }]
        }
    ]
        

    payload = {
        "model": "gpt-4o",
        "messages": message,
        "max_tokens": 1000,
        "temperature": 0
    }


    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Check the response status code
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} for {image_path_pr}")
            print("Response text:", response.text)
            return None

        # Check if the response is empty
        if not response.text:
            print(f"Error: Empty response for {image_path_pr}")
            return None

        try:
            content = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error for {image_path_pr}: {e}")
            print("Response text was:", response.text)
            return None

        return content

    except requests.RequestException as e:
        print(f"Request error for {image_path_pr}: {e}")
        return None
    return content
