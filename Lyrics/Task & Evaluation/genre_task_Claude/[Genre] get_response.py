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
def zero_shot_genre_prompt_gen(lyrics, english_flag):
    if english_flag:
        genre_list_str = "alternative hip hop, alternative pop, alternative r&b, alternative rock, americana, arena rock, bubblegum pop, christmas, country, country pop, country rock, crunk, dance, dance pop, dance rock, dancehall, dirty rap, disco, doo wop, east coast hip hop, edm, electro, electro house, electronic rock, electropop, emo, emo rap, eurodance, folk, folk pop, folk rock, funk, g funk, gangsta rap, glam metal, gospel, hard rock, hip hop, house, indie pop, latin pop, neo soul, new jack swing, new wave, pop, pop punk, pop rap, pop rock, pop soul, post grunge, power pop, r&b, rap rock, reggae, reggae fusion, rock, soft rock, soul, southern hip hop, synth pop, teen pop, trap, tropical house, west coast hip hop, western"
    else:
        genre_list_str = "발라드, 댄스, 랩/힙합, R&B/Soul, 록/메탈, 국내드라마, 인디음악, 포크/블루스, 성인가요/트로트, 국내영화, 일렉트로니카"
    prompt = f"""Here is a list of unique music genres: [{genre_list_str}].\n\n
                Lyrics: '{lyrics}'\n\n
                Say nothing but the Genre as Genre: {{the output}}\n\n
                Output example: Genre: [pop, r&b, hip hop]\n\n
                Genres: """
    return prompt

def cot_genre_prompt_gen(lyrics, english_flag):
    if english_flag:
        genre_list_str = "alternative hip hop, alternative pop, alternative r&b, alternative rock, americana, arena rock, bubblegum pop, christmas, country, country pop, country rock, crunk, dance, dance pop, dance rock, dancehall, dirty rap, disco, doo wop, east coast hip hop, edm, electro, electro house, electronic rock, electropop, emo, emo rap, eurodance, folk, folk pop, folk rock, funk, g funk, gangsta rap, glam metal, gospel, hard rock, hip hop, house, indie pop, latin pop, neo soul, new jack swing, new wave, pop, pop punk, pop rap, pop rock, pop soul, post grunge, power pop, r&b, rap rock, reggae, reggae fusion, rock, soft rock, soul, southern hip hop, synth pop, teen pop, trap, tropical house, west coast hip hop, western"
    else:
        genre_list_str = "발라드, 댄스, 랩/힙합, R&B/Soul, 록/메탈, 국내드라마, 인디음악, 포크/블루스, 성인가요/트로트, 국내영화, 일렉트로니카"

    prompt = f"""Here is a list of unique music genres: [{genre_list_str}].\n\n
                Based on the lyrics provided, identify the genres.\n\n
                Lyrics: '{lyrics}'\n\n
                Say nothing but the Genre as Genre: {{the output}}\n\n
                Output example: Genre: [pop, r&b, hip hop]\n\n
                Genre:"""
    return prompt

def cot_few_shot_genre_prompt_gen(lyrics, english_flag):
    if english_flag:
        genre_list_str = "alternative hip hop, alternative pop, alternative r&b, alternative rock, americana, arena rock, bubblegum pop, christmas, country, country pop, country rock, crunk, dance, dance pop, dance rock, dancehall, dirty rap, disco, doo wop, east coast hip hop, edm, electro, electro house, electronic rock, electropop, emo, emo rap, eurodance, folk, folk pop, folk rock, funk, g funk, gangsta rap, glam metal, gospel, hard rock, hip hop, house, indie pop, latin pop, neo soul, new jack swing, new wave, pop, pop punk, pop rap, pop rock, pop soul, post grunge, power pop, r&b, rap rock, reggae, reggae fusion, rock, soft rock, soul, southern hip hop, synth pop, teen pop, trap, tropical house, west coast hip hop, western"
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
                        """
        example_genres = "pop, soul"
    else:
        genre_list_str = "발라드, 댄스, 랩/힙합, R&B/Soul, 록/메탈, 국내드라마, 인디음악, 포크/블루스, 성인가요/트로트, 국내영화, 일렉트로니카"
        example_lyrics = """
                        그치지 않기를 바랬죠\n
                        처음 그대 내게로 오던 그날에\n
                        잠시 동안 적시는\n
                        그런 비가 아니길\n
                        간절히 난 바래왔었죠\n
                        그대도 내 맘 아나요\n
                        매일 그대만 그려왔던 나를\n
                        오늘도 내 맘에 스며들죠\n
                        그대는 선물입니다\n
                        하늘이 내려준\n
                        홀로 선 세상 속에\n
                        그댈 지켜줄게요\n
                        어느 날 문득\n
                        소나기처럼\n
                        내린 그대지만\n
                        오늘도 불러 봅니다\n
                        내겐 소중한 사람\n
                        Oh\n
                        떨어지는 빗물이\n
                        어느새 날 깨우고\n
                        그대 생각에 잠겨요\n
                        이제는 내게로 와요\n
                        언제나처럼 기다리고 있죠\n
                        그대 손을 꼭 잡아줄게요\n
                        """
        example_genres = "발라드, 국내드라마"
    
    prompt = f"""
            Here is a list of unique music genres: [{genre_list_str}].\n\n

            Example:\n\n
            Lyrics: '{example_lyrics}'\n\n
            Genre: {example_genres}\n\n

            Now, based on the provided lyrics, identify the genres.\n\n
            Lyrics: '{lyrics}'\n\n
            Say nothing but the Genre as Genre: {{the output}}\n\n
            Output example: Genre: [pop, r&b, hip hop]\n\n
            Genre:
            """
    return prompt

def get_response(client, prompt):
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            temperature=0.0,
            max_tokens=40,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    except Exception as e:
        print(e)
        return None

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




### Experiment - ENG
if __name__ == '__main__':
    api_key = "your api key"
    client = anthropic.Anthropic(
        api_key = api_key,
    )
    
    ENG_data_list = read_csv(os.path.join(ROOT_DIR, '01. Original dataset/Billboard_yearly_filtered.csv'))
    ENG_data_retry = read_csv(os.path.join(ROOT_DIR, 'retry_df.csv'))


    with open('Claude_genre_result_yearly_ENG_#14.csv', 'a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Genre', 'Zero-shot', 'COT', 'COT_few_shot'])

        # Get responses
        error_count = 0
        final_result = []

        for ENG_data in tqdm.tqdm(ENG_data_list):
            lyrics = ENG_data['lyrics']

            # prompt generation
            zero_shot_prompt = zero_shot_genre_prompt_gen(lyrics, True)
            cot_prompt = cot_genre_prompt_gen(lyrics, True)
            cot_few_shot_prompt = cot_few_shot_genre_prompt_gen(lyrics, True)

            # get response (Zero-shot)
            zero_shot_flag = True
            zero_shot_response = get_response(client, zero_shot_prompt)
            if (zero_shot_response == None) or zero_shot_response.content[0].text:
                zero_shot_flag = False
            
            # get response (COT)
            if zero_shot_flag:
                continue
            cot_flag = True
            cot_response = get_response(client, cot_prompt)
            if (cot_response == None) or cot_response.content[0].text:
                cot_flag = False

            # get response (COT_few_shot)
            if cot_flag:
                continue
            cot_few_shot_flag = True
            cot_few_shot_response = get_response(client, cot_few_shot_prompt)
            if (cot_few_shot_response == None) or cot_few_shot_response.content[0].text:
                cot_few_shot_flag = False

            # save the result if all responses are valid
            if cot_few_shot_flag:
                continue

        
            try:
                zero_shot_answer = zero_shot_response.content[0].text
                cot_answer = cot_response.content[0].text
                cot_few_shot_answer = cot_few_shot_response.content[0].text

                result = [ENG_data['Title'], ENG_data['Genre'], zero_shot_answer, cot_answer, cot_few_shot_answer]
                writer.writerow(result)

            except Exception as e:
                error_count += 1
                # write error log
                with open(f'error_log_{error_count}.txt', 'a', encoding='utf-8') as file:
                    file.write(f"Error: {e}\n")
                    file.write(f"Original: {ENG_data}\n")
                    file.write(f"Zero-shot: {zero_shot_response}\n")
                    file.write(f"COT: {cot_response}\n")
                    file.write(f"COT_few_shot: {cot_few_shot_response}\n\n")

                continue
            


### Experiment - KOR
if __name__ == '__main__':
    api_key = "your api key"
    client = anthropic.Anthropic(
        api_key = api_key,
    )
    
    KOR_data_list = read_csv(os.path.join(ROOT_DIR, '01. Original dataset./Melon_yearly_filtered.csv'))
    KOR_data_error = KOR_data_list[1700:]

    # Get responses
    with open('Claude_genre_result_yearly_KOR_#3.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Title', 'Genre', 'Zero-shot', 'COT', 'COT_few_shot'])

        error_count = 0
        final_result = []

        for KOR_data in tqdm.tqdm(KOR_data_list):
            lyrics = KOR_data['Lyrics']

            # prompt generation
            zero_shot_prompt = zero_shot_genre_prompt_gen(lyrics, False)
            cot_prompt = cot_genre_prompt_gen(lyrics, False)
            cot_few_shot_prompt = cot_few_shot_genre_prompt_gen(lyrics, False)

            # get response (Zero-shot)
            zero_shot_flag = True
            zero_shot_response = get_response(client, zero_shot_prompt)
            if (zero_shot_response == None) or zero_shot_response.content[0].text:
                zero_shot_flag = False
            
            # get response (COT)
            if zero_shot_flag:
                continue
            cot_flag = True
            cot_response = get_response(client, cot_prompt)
            if (cot_response == None) or cot_response.content[0].text:
                cot_flag = False

            # get response (COT_few_shot)
            if cot_flag:
                continue
            cot_few_shot_flag = True
            cot_few_shot_response = get_response(client, cot_few_shot_prompt)
            if (cot_few_shot_response == None) or cot_few_shot_response.content[0].text:
                cot_few_shot_flag = False

            # save the result if all responses are valid
            if cot_few_shot_flag:
                continue

            try:
                zero_shot_answer = zero_shot_response.content[0].text
                cot_answer = cot_response.content[0].text
                cot_few_shot_answer = cot_few_shot_response.content[0].text

                result = [KOR_data['\ufeffTitle'], KOR_data['Genre'], zero_shot_answer, cot_answer, cot_few_shot_answer]
                writer.writerow(result)

            except Exception as e:
                error_count += 1
                # write error log
                with open(f'error_log_{error_count}.txt', 'a', encoding='utf-8') as file:
                    file.write(f"Error: {e}\n")
                    file.write(f"Original: {KOR_data}\n")
                    file.write(f"Zero-shot: {zero_shot_response}\n")
                    file.write(f"COT: {cot_response}\n")
                    file.write(f"COT_few_shot: {cot_few_shot_response}\n\n")
                
                continue