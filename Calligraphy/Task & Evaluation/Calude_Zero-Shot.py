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


def zero(image_path):
    zero_prmt = "Extract only the Korean characters in the image. Provide no additional information or explanation."

    image1_media_type = get_image_media_type(image_path)  
    image1_data = encode_image(image_path)
    
    client = anthropic.Anthropic(api_key= api_key,)

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
                        "text": zero_prmt
                    }
                ],
            }
        ],
    )
    print(message)
    return message
