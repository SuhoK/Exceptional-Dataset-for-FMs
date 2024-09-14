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

def few(image_path):
    exprompt1="Step1: Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence.\n Identify that the sentence starts with '바라는게' and ends with '안그래? \n\n Step2: Split into words and translate each word in english and identify any typos based on the context.: \n바라는게 (What I hope for) 무한정 (infinitely) 끝없이 (endlessly)내리는 (falling)게 (particle, indicating 'is')아닌게 (is not)엄마나 (Typo: misidentified word, Correct: 얼마나, Translation: how much)다행인지(fortunately)몰라 (I don't know)안그래? (isn't it?)\n\n Step3: Correct the typos by comparing each word with similar words and combine the corrected words to form a coherent sentence.:\n '엄마나' should be '얼마나', '알고 래?' should be '안그래?'\n\nStep4:Combine based on context: \n '바라는게 무한정 끝없이 내리는 게 아닌게 얼마나 다행인지 몰라 안그래?' There is no weird word to use. \n\n Step5: Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one.\n Infer the context: The sentence talks about how fortunate it is that something is not happening endlessly. \n Correct any misrecognized words: '얼 마나' should be '얼마나', '알고 래?' should be '안그래?' \n\n Step6: Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense.\nCheck grammar and spelling: Ensure '바라는게 무한정 끝없이 내리는 게 아닌게 얼마나 다행인지 몰라 안그래?' is grammatically correct and makes sense. \nEnsure the sentence flows naturally and the meaning is clear."
    exprompt2="Step1: Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence. \n Identify that the sentence starts with '해묵은' and ends with '오후 햇살이'. \n\n Step2: Split into words and translate each word in english and identify any typos based on the context.: 해묵은 (aged)나이를 (age)잊은 (forgotten)순간에 (in the moment)따라오는 (following)오후 (afternoon)햇살이 (the sunlight)There is no weird word to use. \n Step3: Correct the typos by comparing each word with similar words and combine the corrected words to form a coherent sentence.:\n There was no weird word \n\nStep4: Combine based on context: '해묵은 나이를 잊은 순간에 따라오는 오후 햇살이' \n\n Step5: Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one. \nInfer the context: The sentence talks about the nostalgic feeling of afternoon sunlight in a moment where old age is forgotten. \n Combine based on context: '해묵은 나이를 잊은 순간에 따라오는 오후 햇살이' \n Correct any misrecognized words: Ensure '해묵은' is correctly recognized and '오후 햇살이' is interpreted correctly \n\n Step6: Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense. Check grammar and spelling: \nEnsure '해묵은 나이를 잊은 순간에 따라오는 오후 햇살이' is grammatically correct and makes sense. \nEnsure the sentence flows naturally and the meaning is clear."
    exprompt3="Step 1: Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence.\nIdentify that the sentence starts with '소중한' and ends with '당신'.\n\nStep 2: Split into words and translate each word into English, and identify any typos based on the context:\n소중한 (Precious, valuable) 당신 (You)\n\nStep 3: Correct the typos by comparing each word with similar words and combine the corrected words to form a coherent sentence:\nNo typos were detected, so the sentence remains '소중한 당신'.\n\nStep 4: Combine based on context:\n'소중한 당신' translates to 'You are precious.' There are no unusual words to correct.\n\nStep 5: Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one.\nThe sentence '소중한 당신' clearly conveys the message 'You are precious.' There are no misrecognized words to correct.\n\nStep 6: Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense.\nCheck grammar and spelling: Ensure '소중한 당신' is grammatically correct and makes sense.\nEnsure the sentence flows naturally and the meaning is clear."
    exprompt4="Step 1: Identify the start and end of the sentence. Check if there are any line breaks in the middle of the sentence.\nIdentify that the sentence starts with '시' and ends with '끝'.\n\nStep 2: Split into words and translate each word into English, and identify any typos based on the context:\n시 (test, exam) 험 (test, exam) 끝 (end, finish)\nThere is no weird word to use.\n\nStep 3: Correct the typos by comparing each word with similar words and combine the corrected words to form a coherent sentence:\nThere were no weird words.\n\nStep 4: Combine based on context: '시험 끝' translates to 'End of the test.'\n\nStep 5: Analyze the context to infer the meaning of the handwriting. Correct any misrecognized words by comparing them with similar words and choosing the correct one.\nInfer the context: The phrase clearly indicates the completion of an exam or test.\nCombine based on context: '시험 끝' translates to 'End of the test.'\nCorrect any misrecognized words: Ensure '시험' is correctly recognized and '끝' is interpreted correctly.\n\nStep 6: Perform grammar and spelling checks to verify the recognized sentence. Ensure that the sentence flows naturally and makes sense. Check grammar and spelling:\nEnsure '시험 끝' is grammatically correct and makes sense.\nEnsure the sentence flows naturally and the meaning is clear."

    media_type = get_image_media_type(image_path)
    image_data = encode_image(image_path)
    
    
    ex_image = encode_image(image_path_ex)
    ex_image2 = encode_image(image_path_ex2)
    ex_image3 = encode_image(image_path_ex3)
    ex_image4 = encode_image(image_path_ex4)
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Below are examples of OCR task. \n I'll show image first and explain step-by-step how to extract text from the image."
                    },
                    {
                        "type": "text",
                        "text": "\n\nFirst Example:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": ex_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": exprompt1
                    },
                    {
                        "type": "text",
                        "text": "\n\nSecond Example:"
                    },
                    
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type2,
                            "data": ex_image2,
                        },
                    },
                    {
                        "type": "text",
                        "text": exprompt2
                    },
                    {
                        "type": "text",
                        "text": "Now, please perform an OCR task on the following image like the example. The image is Korean calligraphy with an illustration. \nTranscribe the letters in the picture with a step-by-step explanation of your reasoning. \n But Don't describe your steps. Just answer the result of the OCR without translating."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract and display only the Korean characters found in the image. Provide no additional information or explanation. Let's think step by step please."
                    }
                ],
            }
        ],
    )
    print(message)
    return message
