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




def gpt_api_cot(image_path):

    prompt="Step1: Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence. \n\n Step2: Split into words and translate each word in english and identify any typos based on the context. \n\n Step3: Correct the typos by comparing each word with similar words and combine the corrected words to form a coherent sentence \n\nStep4: Combine based on context: \n\n Step5: Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one. \nInfer the context: The sentence talks about the nostalgic feeling of afternoon sunlight in a moment where old age is forgotten. \n\n Step6: Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense. Check grammar and spelling."
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    message = [
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text":"The image uploaded is Korean calligraphy with illustration.\n Transcribe the letters in the uploaded image. \nSolve it with following steps. \n1. Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence.\n2. Split the recognized text into individual words. Combine the split words based on the context to form a coherent sentence.\n3. Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one.\n4. Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense \n Don't describe your steps. Just answer the result of the OCR without translating"
            }]
        },
    {
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }]
        }]
        
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
            print(f"Error: Received status code {response.status_code} for {image_path}")
            print("Response text:", response.text)
            return None

        # Check if the response is empty
        if not response.text:
            print(f"Error: Empty response for {image_path}")
            return None

        try:
            content = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error for {image_path}: {e}")
            print("Response text was:", response.text)
            return None

        return content

    except requests.RequestException as e:
        print(f"Request error for {image_path}: {e}")
        return None
    return content
