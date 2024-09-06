import csv
import tqdm as tqdm
import pandas as pd

import anthropic
from anthropic import Anthropic
import os
import requests
import re


ROOT_DIR = "ROOT DIR"


### Prompt Generation
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
			Output example: Description: Honeymoon Avenue is about knowing you are at the end of a relationship and wishing it could not be the end and go back to the beginning and start over. \n\n
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
			Include the possible artist name or song title if relevant.\n\n
			Say nothing but the Description as Description: {{the output}}\n\n
			Output example: Description: Honeymoon Avenue by Ariana Grande is about knowing you are at the end of a relationship and wishing it could not be the end and go back to the beginning and start over. \n\n
			Lyrics: '{lyrics}'\n\n
			Description:
			"""
    return prompt


def get_response(client, prompt):
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        temperature=0.0,
        max_tokens=300,  # max token 설정 => internal error로 이어질 수 있음
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message


def read_csv(path = 'ENG_lyrics_Evaluation_dataset.csv'):
    data = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return data



### Experiment
if __name__ == '__main__':
    api_key = "your api key"
    client = anthropic.Anthropic(
        api_key = api_key,

	# read the data
    data_list = read_csv(os.path.join(ROOT_DIR, '01. Original dataset/Billboard_weekly_filtered.csv'))
    #data_error = data_list[156:]

    with open('Claude_description_weekly_result_#5.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'description', 'Zero-shot', 'COT', 'COT_few_shot'])

        # Get responses
        error_count = 0
        final_result = []
        for data in tqdm.tqdm(data_list):
            lyrics = data['lyrics']

            # prompt generation
            zero_shot_prompt = zero_shot_description_prompt_gen(lyrics)
            cot_prompt = cot_description_prompt_gen(lyrics)
            cot_few_shot_prompt = cot_few_shot_description_prompt_gen(lyrics)
            
            # get response (Zero-shot)
            zero_shot_flag = True
            zero_shot_response = get_response(client, zero_shot_prompt)
            if zero_shot_response.content[0].text:
                zero_shot_flag = False
            
            # get response (COT)
            if zero_shot_flag:
                continue
            cot_flag = True
            cot_response = get_response(client, cot_prompt)
            if cot_response.content[0].text:
                cot_flag = False
            
            # get response (COT_few_shot)
            if cot_flag:
                continue
            cot_few_shot_flag = True
            cot_few_shot_response = get_response(client, cot_few_shot_prompt)
            if cot_few_shot_response.content[0].text:
                cot_few_shot_flag = False

            # save the result if all responses are valid
            if cot_few_shot_flag:
                continue
            
            try:
                zero_shot_answer = zero_shot_response.content[0].text
                cot_answer = cot_response.content[0].text
                cot_few_shot_answer = cot_few_shot_response.content[0].text

                result = [data['Title'], data['description'], zero_shot_answer, cot_answer, cot_few_shot_answer]
                writer.writerow(result)
                
            except Exception as e:
                error_count += 1
                # write error log
                with open(f'error_log_{error_count}.txt', 'a', encoding='utf-8') as file:
                    file.write(f"Error: {e}\n")
                    file.write(f"Original: {data}\n")
                    file.write(f"Zero-shot: {zero_shot_response}\n")
                    file.write(f"COT: {cot_response}\n")
                    file.write(f"COT_few_shot: {cot_few_shot_response}\n\n")