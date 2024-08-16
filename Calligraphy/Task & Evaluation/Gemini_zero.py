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

def gemini_zero(image_path):
    image = image_encoding(image_path)
    prompt = "What are the all Korean characters in the image? Make sure that your answer only includes the result of the OCR without translating. You don't need to describe the processing steps."
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
