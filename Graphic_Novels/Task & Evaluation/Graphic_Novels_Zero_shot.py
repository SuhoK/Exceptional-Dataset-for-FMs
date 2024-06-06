import os
import random
import base64
import requests
import json
import re
import pandas as pd
import ast

# OpenAI API Key
api_key = "Enter your API Key"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to load all image paths from a directory
def load_images(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Path to your image directory
#image_directory = "comics/comics/4_cropped/4_1"
root_directory = "root_directory/images/4_"
df = pd.DataFrame(columns=['No.', 'last_directory', 'Accuracy', 'Suffled image list', 'Result'])   
for num in range(1, 1296):
    image_directory = root_directory + "%d" % num
#    print("image_directory :", image_directory)
    if os.path.exists(image_directory):
        images = load_images(image_directory)
        random.shuffle(images)

        # Check if there is at least one image in the directory
        base64_images = []
        if images:
            # Getting the base64 string of the first image
            for idx in range(len(images)):
                base64_image = encode_image(images[idx])
                base64_images.append(base64_image)

        else:
            print("No images found in the directory.")

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        # Creating the payload
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "The uploaded images represent parts of a story that has been shuffled and consists of 4 images.\
                        Arrange images in the correct order.\
                        \n\n PLEASE!! respond with the list of numbers 1 to 4 in the following format only: [1,2,3,4]\
                        \n\n ONCE AGAIN!!! PLEASE!! respond with the list of numbers 1 to 4 in the following format only: [1,2,3,4]"}]
        }]

        for prompt_image in base64_images:
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prompt_image}"}}]
            }
        )

        payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0,  # Set the temperature to your desired value
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        print(response.json())



        # Extract the value of 'content' and convert it to a list
        content = response_data['choices'][0]['message']['content']
        #number_list = [1,2,3,4]#ast.literal_eval(content)
        number_list = ast.literal_eval(content)

        #print("number_list:", number_list)
        # Print the shuffled image names
        suffled_image_list = []
        for image in images:
            file_name = os.path.basename(image)
    #        print(file_name)
            num_of_image = re.search(r'\d+', file_name).group()
            suffled_image_list.append(int(num_of_image))
        #print("Shuffled images:", suffled_image_list)

        # Result list initialization
        answer_list_4 = [1,3,2,4]
        result = [0] * len(suffled_image_list)

        # Place shuffled elements at new index using gpt_result
        for i, index in enumerate(number_list):
            result[i] = suffled_image_list[index - 1]  # djust index (list starts from 0, so -1)

        # Count the number of elements with the same value at the same index
        A = sum(1 for x, y in zip(result, answer_list_4) if x == y)

        # Calculate Accuracy by dividing by the length of the list
        accuracy = A / len(result)
        #print("result list:", result)
        #print("Same number of numbers at the same index (A):", A)
        #print("Accuracy:", accuracy)

        # image folder name
        last_directory = image_directory.split('/')[-1]
        # add row
        df.loc[len(df)] = [num, last_directory, accuracy, suffled_image_list, result]
        print(df)
    else:
        # If the path does not exist, do nothing (pass over without error).
        continue
# save csv file
df.to_csv('your_directory/Zero-Shot.csv', index=False)

print("done")
