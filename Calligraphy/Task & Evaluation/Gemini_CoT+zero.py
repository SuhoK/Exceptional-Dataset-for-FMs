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

def gemini_cot(image_path):
    image = image_encoding(image_path)
    prompt = "The image uploaded is Korean calligraphy with illustration.\n Transcribe the letters in the uploaded image. Solve it with following steps. \n1. Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence.\n2. Split the recognized text into individual words. Combine the split words based on the context to form a coherent sentence.\n3. Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one.\n4. Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense \n Don't describe your steps. Just answer the result of the OCR without translating"
    combined_input = [prompt, image]
    
    response = model.generate_content(combined_input)
    try:
        content= response.text 
    except:
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason
        print(finish_reason)
        content = None
    
    response = response.to_dict()
    print(content)
    
    return response, content
