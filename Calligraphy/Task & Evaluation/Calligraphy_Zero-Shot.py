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

def gpt_api_zero(image_path):

    # base64 문자열 얻기
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
                "text":"What are the all Korean characters in the image? Make sure that your answer only includes the result of the OCR without translating. You don't need to describe the processing steps."
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

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    content = response.json()
    return(content)
