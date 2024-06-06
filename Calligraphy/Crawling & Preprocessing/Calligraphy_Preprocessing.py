import os
import openai
import requests
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import re
import ctypes
from ctypes import wintypes
import json
import pandas as pd
import numpy as np
from IPython.display import display
import cv2

def remove_special_characters(text):
    return re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]', '', text)

def perform_ocr_on_directory(directory_path, output_file):
    api_key = "Enter yout upstage key"
    url = "https://ap-northeast-2.apistage.ai/v1/document-ai/ocr"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    results = {}
    processed_count = 0  

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(directory_path, filename)
            
            with open(file_path, "rb") as file:
                files = {"image": file}
                response = requests.post(url, headers=headers, files=files)
                
                if response.status_code == 200:
                    results[filename] = response.json()
                    text_results = [item['text'] for page in results[filename]['pages'] for item in page['words']]
                    print(f"Processed: {filename} - Texts: {text_results}")
                else:
                    results[filename] = "Error: " + response.text
                    print(f"Processed: {filename} - Error: {response.text}")
                
                processed_count += 1
                print(f"Completed {processed_count} files so far.")
                time.sleep(1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



# JSON 파일과 이미지 경로 설정
json_path = r"C:\Users\user\Desktop\lab\paper\ocr_result.json"
image_dir = r"C:\Users\user\Desktop\lab\paper\confound"
output_dir = r"C:\Users\user\Desktop\lab\paper\output"

# 바운딩 박스 패딩 (픽셀 단위)
padding = 10

# JSON 파일 로드
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


def get_padded_bbox(vertices, padding, img_width, img_height):
    min_x = min(vertex['x'] for vertex in vertices)
    min_y = min(vertex['y'] for vertex in vertices)
    max_x = max(vertex['x'] for vertex in vertices)
    max_y = max(vertex['y'] for vertex in vertices)
    
    min_x = max(min_x - padding, 0)
    min_y = max(min_y - padding, 0)
    max_x = min(max_x + padding, img_width)
    max_y = min(max_y + padding, img_height)
    
    return (min_x, min_y, max_x, max_y)

def cropped(json_path,image_dir,output_dir)
    for image_name, image_data in data.items():
        image_path = os.path.join(image_dir, image_name)
        img_array = np.fromfile(image_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_height, img_width, _ = image.shape
        words = image_data['pages'][0]['words']
        crops = []
    
        for word in words:
            vertices = word['boundingBox']['vertices']
            bbox = get_padded_bbox(vertices, padding, img_width, img_height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                print(f"invalid: {bbox}")
                continue
    
            crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
            if crop.size == 0:
                print(f"empty: {bbox}")
                continue
    
            crops.append((crop, bbox))

        base_name = os.path.splitext(image_name)[0]
        for i, (crop, bbox) in enumerate(crops):
            crop_path = os.path.join(output_dir, f"{base_name}_cropped_image_{i}.png")
            os.makedirs(output_dir, exist_ok=True)
            success = cv2.imwrite(crop_path, crop)

