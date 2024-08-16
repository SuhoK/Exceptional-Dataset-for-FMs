import csv
import tqdm
import google.auth
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.urllib3 import AuthorizedHttp
import google.auth.transport.requests
from google.api_core import client_options as client_options_lib
import google.generativeai as genai


# Description
def zero_shot_description_prompt_gen(lyrics):
    prompt = f"""Say nothing but the Description as Description: {{the output}}\n\n
Output example: Description: The song explores themes of love and heartbreak.\n\n
Lyrics: '{lyrics}'\n\n
Description:"""
    return prompt

def cot_description_prompt_gen(lyrics):
    prompt = f"""Based on the lyrics provided, write a brief description of the song.\n\n
Say nothing but the Description as Description: {{the output}}\n\n
Output example: Description: The song is about knowing you are at the end of a relationship and wishing it could not be the end and go back to the beginning and start over. \n\n
Lyrics: '{lyrics}'\n\n
Description:"""
    return prompt

def cot_few_shot_description_prompt_gen(lyrics):

    example_lyrics = """
I'd like to say we gave it a try\n
I'd like to blame it all on life\n
Maybe we just weren't right\n
But that's a lie, that's a lie\n\n

And we can deny it as much as we want\n
But in time, our feelings will show\n
'Cause sooner or later, we'll wonder why we gave up\n
The truth is everyone knows, oh\n\n

Almost, almost is never enough\n
So close to being in love\n
If I would have known that you wanted me the way I wanted you\n
Then maybe we wouldn't be two worlds apart (Ah)\n
But right here in each other's arms\n
And we almost, we almost knew what love was\n
But almost is never enough (Ah)\n\n

"""

    example_description = """
On the collaborative track “Almost Is Never Enough,” Ariana Grande & Nathan Sykes play a couple who had a relationship that hadn’t gone right. 
Ariana would like to say things were going well but she knows that’s a lie and like the title states, almost is never enough to make the relationship work; you need to put full effort in. 
Both of them state that they didn’t feel the relationship while in it, but the mood of the song and lyrics suggest that they both want to either reconnect or they simply just miss better times.\n\n
At the time of the song’s release, Nathan and Ariana were dating. Unfortunately, their relationship ended a few months later.
"""

    prompt = f"""
Example:\n\n
Lyrics: '{example_lyrics}'\n\n
Genre: {example_description}\n\n

Based on the provided lyrics, write a brief description of the song.\n\n
Say nothing but the Description as Description: {{the output}}\n\n
Output example: Description: Honeymoon Avenue is about knowing you are at the end of a relationship and wishing it could not be the end and go back to the beginning and start over. \n\n
Lyrics: '{lyrics}'\n\n
Description:
"""
    return prompt

def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

def description():
    # set up the model
    safe = [
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH",
        },
    ]
    genai.configure(api_key="AIzaSyAgLnMIqKYQPDBGUjlTKHaIZ016q8wq3b4")
    client_options = client_options_lib.ClientOptions(api_endpoint="us-east1-generativelanguage.googleapis.com") 

    generation_config = {
        "temperature": 0.01
    }
    model = genai.GenerativeModel(model_name="gemini-1.5-pro", 
                                generation_config=generation_config,
                                safety_settings=safe)

    # read the data
    data_list = read_csv('Billboard_weekly_filtered.csv')

    # Get responses
    count = 0
    final_result = []
    for data in tqdm.tqdm(data_list):
        lyrics = data['lyrics']

        # prompt generation
        zero_shot_prompt = zero_shot_description_prompt_gen(lyrics)
        cot_prompt = cot_description_prompt_gen(lyrics)
        cot_few_shot_prompt = cot_few_shot_description_prompt_gen(lyrics)
        
        # get response (Zero-shot)
        zero_shot_flag = True
        zero_shot_response = model.generate_content([zero_shot_prompt])
        if zero_shot_response.candidates[0].content:
            zero_shot_flag = False
        
        # get response (COT)
        if zero_shot_flag:
            continue
        cot_flag = True
        cot_response = model.generate_content([cot_prompt])
        if cot_response.candidates[0].content:
            cot_flag = False
        
        # get response (COT_few_shot)
        if cot_flag:
            continue
        cot_few_shot_flag = True
        cot_few_shot_response = model.generate_content([cot_few_shot_prompt])
        if cot_few_shot_response.candidates[0].content:
            cot_few_shot_flag = False

        # save the result if all responses are valid
        if cot_few_shot_flag:
            continue

        result = [data['Title'], data['description'], zero_shot_response.candidates[0].content, cot_response.candidates[0].content, cot_few_shot_response.candidates[0].content]
        final_result.append(result)
    print(len(final_result))

    # write on the csv file
    with open('description_result_#3.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Description', 'Zero-shot', 'COT', 'COT_few_shot'])
        writer.writerows(final_result)

if __name__ == "__main__":
    description()