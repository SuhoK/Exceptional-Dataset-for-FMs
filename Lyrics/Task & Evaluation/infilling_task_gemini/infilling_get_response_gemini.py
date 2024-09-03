import csv
import tqdm as tqdm

import google.generativeai as genai
from google.api_core import client_options as client_options_lib

def read_csv(path = 'ENG_lyrics_Evaluation_dataset.csv'):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


# Infilling
## Prompt generation
def zero_shot_prompt_gen(lyrics):
    prompt = f"""You are a powerful language model. Fill in the blanks in the following text with appropriate words. "
The text is a part of a song with certain words masked by [MASK].\n\n"
Lyrics: '{lyrics}'\n\n"
Say nothing but the filled lyrics as 'Filled lyrics: {{the output}}'.\n\n"
Output example: Filled lyrics: 'I know this pain (I know this pain) why do you lock yourself up in these chains? (these chains)...'"""

    return prompt

def cot_prompt_gen(lyrics):
    prompt = f"""You are a powerful language model. Fill in the blanks in the following text with appropriate words.
The text is a part of a song with certain words masked by [MASK]. For each blank, think step by step about the context and meaning of the surrounding text before choosing the word.
To do this, follow these steps:
    a. Carefully read and analysis the lyrics.
    b-1. Check the entire lyrics to see if there are any repeating parts.
    b-2. If repeating parts exist, replace the [MASK] with the corresponding word.
    c-1. Make the list of possible words for the masked part.
    c-2. Select a suitable word from the candidate list.
    c-3. Replace [MASK] with the word that you selected.
Lyrics: '{lyrics}'
Step-by-step reasoning and filled lyrics as 'Filled lyrics: {{the output}}'.
Say nothing but the filled lyrics as 'Filled lyrics: {{the output}}'.
Output example: Filled lyrics: 'I know this pain (I know this pain) why do you lock yourself up in these chains? (these chains)...'
"""

    return prompt

def cot_few_shot_prompt_gen(lyrics, english_flag):
    if english_flag:
        example_lyrics = """Rotgut whiskey's gonna ease my mind Beach [MASK] rests on the dryin' line Do I remind you of your daddy in his '88 Ford? Labrador [MASK] out the passenger door The sand from your hair is blowin' in my eyes [MASK] it on [MASK] [MASK] grown men don't cry [MASK] [MASK] remember that beat down basement couch? I'd sing [MASK] my love songs [MASK] you'd tell me about How your mama [MASK] off and pawned her ring [MASK] remember, I remember everything 
A cold shoulder at closing time You were bеggin' [MASK] to stay 'til the sun [MASK] Strangе words come on out Of a [MASK] man's mouth when his [MASK] broke Pictures and passin' time You only [MASK] like that when you're drinkin' [MASK] wish I didn't, but I do Remember every moment [MASK] the nights with you  
You're drinkin' everything to ease your mind But when the hell are you gonna ease mine? You're like concrete feet in the summer heat [MASK] burns like hell when [MASK] souls meet No, you'll never be the man that you always swore [MASK] [MASK] remember you singin' [MASK] that '88 [MASK]  
A cold shoulder at closing [MASK] You were beggin' me to stay 'til the sun rose Strange words [MASK] on out Of a [MASK] [MASK] mouth when his mind's broke Pictures and passin' time You only smile like that when you're drinkin' I wish I didn't, but I do Remember every moment on [MASK] nights with you Cold shoulder at closing time You were beggin' me [MASK] stay 'til the sun rose Strange words come on out Of a grown man's mouth when his mind's broke Pictures and passin' time You only [MASK] [MASK] that when you're drinkin' I wish I didn't, but I do Remember every moment on the nights with you  """

        example_filled = """Rotgut whiskey's gonna ease my mind Beach towel rests on the dryin' line Do I remind you of your daddy in his '88 Ford? Labrador hangin' out the passenger door The sand from your hair is blowin' in my eyes Blame it on the beach, grown men don't cry Do you remember that beat down basement couch? I'd sing you my love songs and you'd tell me about How your mama ran off and pawned her ring I remember, I remember everything 
A cold shoulder at closing time You were bеggin' me to stay 'til the sun rose Strangе words come on out Of a grown man's mouth when his mind's broke Pictures and passin' time You only smile like that when you're drinkin' I wish I didn't, but I do Remember every moment on the nights with you 
You're drinkin' everything to ease your mind But when the hell are you gonna ease mine? You're like concrete feet in the summer heat It burns like hell when two souls meet No, you'll never be the man that you always swore But I'll remember you singin' in that '88 Ford 
A cold shoulder at closing time You were beggin' me to stay 'til the sun rose Strange words come on out Of a grown man's mouth when his mind's broke Pictures and passin' time You only smile like that when you're drinkin' I wish I didn't, but I do Remember every moment on the nights with you Cold shoulder at closing time You were beggin' me to stay 'til the sun rose Strange words come on out Of a grown man's mouth when his mind's broke Pictures and passin' time You only smile like that when you're drinkin' I wish I didn't, but I do Remember every moment on the nights with you """
    
    else:
        example_lyrics = """세상에 음악의 신이 있다면 고맙다고 안아주고 싶어 전 세계 공통의 Language 자음과 모음이 달라도 상관없는 건 Music
            말이 안 통해도 [MASK] 있다면 [MASK] 지금부터는 아주 친한 친구 너와 내가 모르는 사이여도 춤출 [MASK] 있어 We [MASK] mix it up right
            Sugar and spice Brass sound and guitar 네 [MASK] 다 내 [MASK] 쿵치팍치 또한 내 이름인가
            이것 또한 나를 위한 소린가 [MASK] [MASK] Drum bass Piano [MASK]
            무엇이 우리의 행복인가 뭐 있나 춤을 춰 노래하자 [MASK] 우리의 행복이다
            [MASK] 한번 더 Hey 음악은 우리의 숨이니까 위험하지 않아 계속 들이키자 Everybody 귀를 기울여 보자
            세상에 음악의 신이 있다면 이건 당신께 주는 메시지 음정 하나하나 모아보자 음- 춤춰 노래해 기분이 끝내주네
            쿵 치 팍 [MASK] 쿵 쿵 치 팍 [MASK] 예 쿵 치 팍 치 [MASK] [MASK] 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 [MASK] 예 행복은 바로 지금이야
            생각해 봐 우리는 소음마저 음악이야 저마다의 [MASK] 맞춰가며 살아가 개미의 발소리마저도 Harmony
            [MASK] [MASK] 행복인가 뭐 있나 춤을 춰 노래하자 이것이 우리의 행복이다 다시 한번 더 [MASK]
            음악은 우리의 숨이니까 [MASK] 않아 계속 들이키자 Everybody 귀를 기울여 보자
            세상에 [MASK] 신이 있다면 이건 당신께 주는 메시지 음정 하나하나 모아보자 음- 춤춰 노래해 기분이 끝내주네
            [MASK] 치 팍 치 쿵 [MASK] 치 팍 치 예 [MASK] 치 팍 치 [MASK] 쿵 치 팍 치 예 [MASK] 치 팍 치 쿵 쿵 치 팍 치 예 음악의 신이 있다면
            SEVENTEEN to the world 전 세계 다 합창 쿵 치 [MASK] 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 [MASK] 쿵 치 [MASK] 치 예
            [MASK] 치 팍 치 쿵 [MASK] 치 팍 치 [MASK] 음악의 신이 있다면"""

        example_filled = """세상에 음악의 신이 있다면 고맙다고 안아주고 싶어  전 세계 공통의 Language 자음과 모음이 달라도 상관없는 건 Music
            말이 안 통해도 음악이 있다면 우리는 지금부터는 아주 친한 친구  너와 내가 모르는 사이여도 춤출 수 있어 We can mix it up right
            Sugar and spice Brass sound and guitar  네 글자면 다 내 이름이래 쿵치팍치 또한 내 이름인가
            이것 또한 나를 위한 소린가 Kick snare Drum bass Piano Bassline
            무엇이 우리의 행복인가 뭐 있나 춤을 춰 노래하자 이것이 우리의 행복이다 
            다시 한번 더 Hey  음악은 우리의 숨이니까 위험하지 않아 계속 들이키자 Everybody 귀를 기울여 보자
            세상에 음악의 신이 있다면 이건 당신께 주는 메시지 음정 하나하나 모아보자 음- 춤춰 노래해 기분이 끝내주네
            쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예 행복은 바로 지금이야
            생각해 봐 우리는 소음마저 음악이야 저마다의 쿵짝 맞춰가며 살아가 개미의 발소리마저도 Harmony
            무엇이 우리의 행복인가 뭐 있나 춤을 춰 노래하자 이것이 우리의 행복이다  다시 한번 더 Hey
            음악은 우리의 숨이니까 위험하지 않아 계속 들이키자 Everybody 귀를 기울여 보자 
            세상에 음악의 신이 있다면 이건 당신께 주는 메시지 음정 하나하나 모아보자 음- 춤춰 노래해 기분이 끝내주네 
            쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예 음악의 신이 있다면 
            SEVENTEEN to the world 전 세계 다 합창  쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예
            쿵 치 팍 치 쿵 쿵 치 팍 치 예 음악의 신이 있다면"""

    prompt = f"""You are a powerful language model. Fill in the blanks in the following text with appropriate words.
The text is a part of a song with certain words masked by [MASK]. For each blank, think step by step about the context and meaning of the surrounding text before choosing the word.
To do this, follow these steps:
    a. Carefully read and analysis the lyrics.
    b-1. Check the entire lyrics to see if there are any repeating parts.
    b-2. If repeating parts exist, replace the [MASK] with the corresponding word.
    c-1. Make the list of possible words for the masked part.
    c-2. Select a suitable word from the candidate list.
    c-3. Replace [MASK] with the word that you selected.
Example:
Lyrics: '{example_lyrics}'
Filled lyrics: '{example_filled}'
Now, based on the provided lyrics, fill in the blanks with appropriate words.
Lyrics: '{lyrics}'
Step-by-step reasoning and filled lyrics as 'Filled lyrics: {{the output}}'.
Say nothing but the filled lyrics as 'Filled lyrics: {{the output}}'.
Output example: Filled lyrics: 'I know this pain (I know this pain) why do you lock yourself up in these chains? (these chains)...'
"""

    return prompt

## Get response
def safe_setting():
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
    return safe

def infilling_Eng(api_key, data_list):
    safe = safe_setting()

    genai.configure(api_key=api_key)
    client_options = client_options_lib.ClientOptions(api_endpoint="us-east1-generativelanguage.googleapis.com") 

    generation_config = {
        "temperature": 0.01
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-pro", 
                                generation_config=generation_config,
                                safety_settings=safe)


    ENG_data_list = read_csv(data_list)

    final_result = []

    for ENG_data in tqdm.tqdm(ENG_data_list):
        masked_lyrics = ENG_data['Original']

        # prompt generation
        zero_shot_prompt = zero_shot_prompt_gen(masked_lyrics)
        COT_prompt = cot_prompt_gen(masked_lyrics)
        # True: English, False: Korean
        COT_few_shot_prompt = cot_few_shot_prompt_gen(masked_lyrics, True)

        try:
            # get response (Zero-shot)
            zero_shot_flag = True
            zero_shot_response = model.generate_content([zero_shot_prompt])
            if zero_shot_response.candidates[0].content:
                zero_shot_flag = False

            # get response (COT)
            if zero_shot_flag:
                continue
            COT_flag = True
            COT_response = model.generate_content([COT_prompt])
            if COT_response.candidates[0].content:
                COT_flag = False
            
            # get response (COT_few_shot)
            if COT_flag:
                continue
            COT_few_shot_flag = True
            COT_few_shot_response = model.generate_content([COT_few_shot_prompt])
            if COT_few_shot_response.candidates[0].content:
                COT_few_shot_flag = False
            
            # save the result if all responses are valid
            if COT_few_shot_flag:
                continue

            result =  [ENG_data['Title'], ENG_data['Answer'], zero_shot_response.text, COT_response.text, COT_few_shot_response.text]
            final_result.append(result)
        except Exception as error:
            print(error)

    print(len(final_result))

    with open('Gemini_infilling_result_ENG.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Original', 'Zero-shot', 'COT', 'COT_few_shot'])
        writer.writerows(final_result)

def infilling_KOR(api_key, data_list):
    safe = safe_setting()

    genai.configure(api_key=api_key)
    client_options = client_options_lib.ClientOptions(api_endpoint="us-east1-generativelanguage.googleapis.com") 

    generation_config = {
        "temperature": 0.01
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-pro", 
                                generation_config=generation_config,
                                safety_settings=safe)


    KOR_data_list = read_csv(data_list)
    KOR_answer_list = read_csv(data_list)

    final_result = []

    for i in tqdm.tqdm(range(len(KOR_data_list))):
        ENG_data = KOR_data_list[i]
        masked_lyrics = ENG_data['Original']

        # prompt generation
        zero_shot_prompt = zero_shot_prompt_gen(masked_lyrics)
        COT_prompt = cot_prompt_gen(masked_lyrics)
        # True: English, False: Korean
        COT_few_shot_prompt = cot_few_shot_prompt_gen(masked_lyrics, False)

        try:
            # get response (Zero-shot)
            zero_shot_flag = True
            zero_shot_response = model.generate_content([zero_shot_prompt])
            if zero_shot_response.candidates[0].content:
                zero_shot_flag = False

            # get response (COT)
            if zero_shot_flag:
                continue
            COT_flag = True
            COT_response = model.generate_content([COT_prompt])
            if COT_response.candidates[0].content:
                COT_flag = False
            
            # get response (COT_few_shot)
            if COT_flag:
                continue
            COT_few_shot_flag = True
            COT_few_shot_response = model.generate_content([COT_few_shot_prompt])
            if COT_few_shot_response.candidates[0].content:
                COT_few_shot_flag = False
            
            # save the result if all responses are valid
            if COT_few_shot_flag:
                continue

            result =  [ENG_data['Title'], ENG_data['Original'], KOR_answer_list[i]['Original'], zero_shot_response.text, COT_response.text, COT_few_shot_response.text]
            final_result.append(result)
        except Exception as error:
            print(error)
    print(len(final_result))

    with open('Gemini_infilling_result_KOR.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Masked', 'Original', 'Zero-shot', 'COT', 'COT_few_shot'])
        writer.writerows(final_result)

if __name__ == '__main__':
    infilling_Eng("YOUR_API_KEY")
    infilling_KOR("YOUR_API_KEY")
