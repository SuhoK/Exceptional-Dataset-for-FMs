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

genai.configure(api_key="Enter your API Key")  
client_options = client_options_lib.ClientOptions(api_endpoint="us-east1-generativelanguage.googleapis.com") 
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

def image_encoding(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
        return image
    except (IOError, OSError) as e:
        print(f"Error loading image '{image_path}': {e}")
        return None

def gemini_cot_few(image_path):
    image_path_ex1 = "example_image1"
    image_path_ex2 = "example_image2"

    ex_image1 = image_encoding(image_path_ex1)
    ex_image2 = image_encoding(image_path_ex2)
    
    image = image_encoding(image_path)
    
    intro = "Below are examples of OCR tasks. Please transcribe the letters in the following images."
    
    exprompt1 = (
        "### Example 1:\n\n"
        "**Step 1:** Identify the start and end of the sentence. The sentence starts with '바라는게' and ends with '안그래?'\n"
        "**Step 2:** Split into words: 바라는게, 무한정, 끝없이, 내리는, 게, 아닌게, 얼마나, 다행인지, 몰라, 안그래?\n"
        "**Step 3:** Correct typos and combine words: '얼마나' instead of '엄마나', '안그래?' instead of '알고 래?'\n"
        "**Step 4:** Combine and correct context: '바라는게 무한정 끝없이 내리는 게 아닌게 얼마나 다행인지 몰라 안그래?'\n"
    )
    
    exprompt2 = (
        "### Example 2:\n\n"
        "**Step 1:** Identify the start and end of the sentence. The sentence starts with '해묵은' and ends with '오후 햇살이'\n"
        "**Step 2:** Split into words: 해묵은, 나이를, 잊은, 순간에, 따라오는, 오후, 햇살이\n"
        "**Step 3:** Combine based on context: '해묵은 나이를 잊은 순간에 따라오는 오후 햇살이'\n"
    )
    
    prompt = (
        "Now, please perform an OCR task on the following image like the examples. The image is Korean calligraphy with an illustration.\n"
        "Transcribe the letters in the picture with a step-by-step explanation of your reasoning. But don't describe your steps. Just provide the result of the OCR without translating. Now let's think step by step."
    )
    
    combined_input = [intro, exprompt1, ex_image1, exprompt2, ex_image2, prompt, image]
    
    response = model.generate_content(combined_input)
    
    try:
        content = response.text 
    except:
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason
        print(finish_reason)
        content = None
    
    response = response.to_dict()
    print(content)
    
    return response, content

