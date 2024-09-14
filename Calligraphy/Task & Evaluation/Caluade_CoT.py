!pip install anthropic

import anthropic
import base64
import os
import csv
from tqdm import tqdm
import time
import os
import random
import base64
import json
import re
import pandas as pd
import ast
import google.generativeai as genai
from google.api_core import client_options as client_options_lib
from PIL import Image
import io
import csv

api_key="Enter your key"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_media_type(image_path):
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    if ext == ".png":
        return "image/png"
    elif ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    elif ext == ".gif":
        return "image/gif"
    else:
        raise ValueError(f"Unsupported image format: {ext}")


def cot(image_path):
    image1_media_type = get_image_media_type(image_path) 
    image1_data = encode_image(image_path)
    cot_prmt = "The image uploaded is Korean calligraphy with illustration.\n Transcribe the letters in the uploaded image. \nSolve it with following steps. \n1. Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence.\n2. Split the recognized text into individual words. Combine the split words based on the context to form a coherent sentence.\n3. Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one.\n4. Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense \n Provide no additional information or explanation.\nJust answer the result of the OCR without translating."
   
    client = anthropic.Anthropic(api_key=api_key,)

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image1_media_type,
                            "data": image1_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": cot_prmt
                    }
                ],
            }
        ],
    )
    print(message)
    return message
