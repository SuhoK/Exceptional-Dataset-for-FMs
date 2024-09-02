import csv
import tqdm
import os
import google.generativeai as genai
from google.api_core import client_options as client_options_lib

def read_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# Genre
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
If I would have known that you wanted me the way I wanted you\n
Then maybe we wouldn't be two worlds apart (Ah)\n
But right here in each other's arms\n
And we almost, we almost knew what love was\n
But almost is never enough (Ah)\n\n
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

def genre_ENG(start_index=0):
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
    genai.configure(api_key="your api key")
    client_options = client_options_lib.ClientOptions(api_endpoint="us-east1-generativelanguage.googleapis.com") 

    generation_config = {
        "temperature": 0.01
    }
    model = genai.GenerativeModel(model_name="gemini-1.5-pro", 
                                generation_config=generation_config,
                                safety_settings=safe)
    model.output_token_limit=40

    # read the data
    data_list = read_csv('Billboard_yearly_filtered.csv')

    # Get responses
    final_result = []
    for index, data in enumerate(tqdm.tqdm(data_list[start_index:]), start=start_index):
        try:
            if index % 200 == 0:
                # write on the csv file
                file_exists = os.path.isfile('description_result_#1_Yearly.csv')
                with open(f'Gemini_genre_result_ENG_yearly_#1_{index}.csv', 'a', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(['Title', 'Genre', 'Zero-shot', 'COT', 'COT_few_shot'])
                    writer.writerows(final_result)
            lyrics = data['lyrics']
    
            # prompt generation
            zero_shot_prompt = zero_shot_genre_prompt_gen(lyrics, True)
            cot_prompt = cot_genre_prompt_gen(lyrics, True)
            cot_few_shot_prompt = cot_few_shot_genre_prompt_gen(lyrics, True)
    
            # get response (Zero-shot)
            zero_shot_response = model.generate_content([zero_shot_prompt])
            if not zero_shot_response.candidates or not zero_shot_response.candidates[0].content:
                continue
            
            # get response (COT)
            cot_response = model.generate_content([cot_prompt])
            if not cot_response.candidates or not cot_response.candidates[0].content:
                continue
            
            # get response (COT_few_shot)
            cot_few_shot_response = model.generate_content([cot_few_shot_prompt])
            if not cot_few_shot_response.candidates or not cot_few_shot_response.candidates[0].content:
                continue
    
            result = [data['Title'], data['Genre'], zero_shot_response.candidates[0].content, cot_response.candidates[0].content, cot_few_shot_response.candidates[0].content]
            final_result.append(result)
        except Exception as e:
            print("Something bad happened :(")
            print(e)
            continue
    print(len(final_result))

    # write on the csv file
    file_exists = os.path.isfile('Gemini_genre_result_ENG_yearly_#1.csv')
    with open('Gemini_genre_result_ENG_yearly_#1.csv', 'a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Title', 'Genre', 'Zero-shot', 'COT', 'COT_few_shot'])
        writer.writerows(final_result)

def genre_KOR(start_index=0):
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
    genai.configure(api_key="your api key")
    client_options = client_options_lib.ClientOptions(api_endpoint="us-east1-generativelanguage.googleapis.com") 

    generation_config = {
        "temperature": 0.01
    }
    model = genai.GenerativeModel(model_name="gemini-1.5-pro", 
                                generation_config=generation_config,
                                safety_settings=safe)
    model.output_token_limit=40

    # read the data
    data_list = read_csv('[1990-2023 Lyrics, Genre, Description] Melon.csv')

    # Get responses
    final_result = []
    for index, data in enumerate(tqdm.tqdm(data_list[start_index:]), start=start_index):
        try:
            if index % 200 == 0:
                # write on the csv file
                file_exists = os.path.isfile('description_result_#1_Yearly.csv')
                with open(f'Gemini_genre_result_KOR_yearly_#1_{index}.csv', 'a', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(['Title', 'Genre', 'Zero-shot', 'COT', 'COT_few_shot'])
                    writer.writerows(final_result)
            lyrics = data['Lyrics']
    
            # prompt generation
            zero_shot_prompt = zero_shot_genre_prompt_gen(lyrics, False)
            cot_prompt = cot_genre_prompt_gen(lyrics, False)
            cot_few_shot_prompt = cot_few_shot_genre_prompt_gen(lyrics, False)
    
            # get response (Zero-shot)
            zero_shot_response = model.generate_content([zero_shot_prompt])
            if not zero_shot_response.candidates or not zero_shot_response.candidates[0].content:
                continue
            
            # get response (COT)
            cot_response = model.generate_content([cot_prompt])
            if not cot_response.candidates or not cot_response.candidates[0].content:
                continue
            
            # get response (COT_few_shot)
            cot_few_shot_response = model.generate_content([cot_few_shot_prompt])
            if not cot_few_shot_response.candidates or not cot_few_shot_response.candidates[0].content:
                continue
    
            result = [data['Title'], data['Genre'], zero_shot_response.candidates[0].content, cot_response.candidates[0].content, cot_few_shot_response.candidates[0].content]
            final_result.append(result)

        except Exception as e:
            print("Something bad happened :(")
            print(e)
            continue
    print(len(final_result))

    # write on the csv file
    file_exists = os.path.isfile('Gemini_genre_result_KOR_yearly_#1.csv')
    with open('Gemini_genre_result_KOR_yearly_#1.csv', 'a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Title', 'Genre', 'Zero-shot', 'COT', 'COT_few_shot'])
        writer.writerows(final_result)

if __name__ == '__main__':
    genre_ENG(start_index=0)