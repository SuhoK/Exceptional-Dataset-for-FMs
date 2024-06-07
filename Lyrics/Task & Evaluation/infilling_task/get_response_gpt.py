import pandas as pd
import requests
from openai import OpenAI
import json
import csv

def make_request(payload, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    return response_data

def prepare_zero_shot_payload(lyrics):
    prompt = (
        f"You are a powerful language model. Fill in the blanks in the following text with appropriate words. "
        f"The text is a part of a song with certain words masked by [MASK].\n\n"
        f"Lyrics: '{lyrics}'\n\n"
        f"Say nothing but the filled lyrics as 'Filled lyrics: {{the output}}'.\n\n"
        f"Output example: Filled lyrics: 'I know this pain (I know this pain) why do you lock yourself up in these chains? (these chains)...'"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.0
    }
    
    return payload

def prepare_cot_payload(lyrics):
    prompt = (
        f"You are a powerful language model. Fill in the blanks in the following text with appropriate words. "
        f"The text is a part of a song with certain words masked by [MASK]. For each blank, think step by step \
        about the context and meaning of the surrounding text before choosing the word.\n\n"
        f"To do this, follow these steps:\n"
        f"  a. Carefully read and analysis the lyrics.\n"
        f"  b-1. Check the entire lyrics to see if there are any repeating parts.\n"
        f"  b-2. If repeating parts exist, replace the [MASK] with the corresponding word.\n"
        f"  c-1. Make the list of possible words for the masked part.\n"
        f"  c-2. Select a suitable word from the candidate list.\n"
        f"  c-3. Replace [MASK] with the word that you selected.\n"
        f"Lyrics: '{lyrics}'\n\n"
        f"Step-by-step reasoning and filled lyrics as 'Filled lyrics: {{the output}}'.\n\n"
        f"Say nothing but the filled lyrics as 'Filled lyrics: {{the output}}'.\n\n"
        f"Output example: Filled lyrics: 'I know this pain (I know this pain) why do you lock yourself up in these chains? (these chains)...'"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.0
    }
    
    return payload

def prepare_cot_few_shot_payload(lyrics):
    example_lyrics = (
            "Rotgut whiskey's gonna ease my mind Beach [MASK] rests on the dryin' line Do I remind you of your daddy in his '88 Ford? Labrador [MASK] out the passenger door The sand from your hair is blowin' in my eyes [MASK] it on [MASK] [MASK] grown men don't cry [MASK] [MASK] remember that beat down basement couch? I'd sing [MASK] my love songs [MASK] you'd tell me about How your mama [MASK] off and pawned her ring [MASK] remember, I remember everything "
            "A cold shoulder at closing time You were bеggin' [MASK] to stay 'til the sun [MASK] Strangе words come on out Of a [MASK] man's mouth when his [MASK] broke Pictures and passin' time You only [MASK] like that when you're drinkin' [MASK] wish I didn't, but I do Remember every moment [MASK] the nights with you  "
            "You're drinkin' everything to ease your mind But when the hell are you gonna ease mine? You're like concrete feet in the summer heat [MASK] burns like hell when [MASK] souls meet No, you'll never be the man that you always swore [MASK] [MASK] remember you singin' [MASK] that '88 [MASK]  "
            "A cold shoulder at closing [MASK] You were beggin' me to stay 'til the sun rose Strange words [MASK] on out Of a [MASK] [MASK] mouth when his mind's broke Pictures and passin' time You only smile like that when you're drinkin' I wish I didn't, but I do Remember every moment on [MASK] nights with you Cold shoulder at closing time You were beggin' me [MASK] stay 'til the sun rose Strange words come on out Of a grown man's mouth when his mind's broke Pictures and passin' time You only [MASK] [MASK] that when you're drinkin' I wish I didn't, but I do Remember every moment on the nights with you  "
    )
    example_filled = (
            "Rotgut whiskey's gonna ease my mind Beach towel rests on the dryin' line Do I remind you of your daddy in his '88 Ford? Labrador hangin' out the passenger door The sand from your hair is blowin' in my eyes Blame it on the beach, grown men don't cry Do you remember that beat down basement couch? I'd sing you my love songs and you'd tell me about How your mama ran off and pawned her ring I remember, I remember everything "
            "A cold shoulder at closing time You were bеggin' me to stay 'til the sun rose Strangе words come on out Of a grown man's mouth when his mind's broke Pictures and passin' time You only smile like that when you're drinkin' I wish I didn't, but I do Remember every moment on the nights with you "
            "You're drinkin' everything to ease your mind But when the hell are you gonna ease mine? You're like concrete feet in the summer heat It burns like hell when two souls meet No, you'll never be the man that you always swore But I'll remember you singin' in that '88 Ford "
            "A cold shoulder at closing time You were beggin' me to stay 'til the sun rose Strange words come on out Of a grown man's mouth when his mind's broke Pictures and passin' time You only smile like that when you're drinkin' I wish I didn't, but I do Remember every moment on the nights with you Cold shoulder at closing time You were beggin' me to stay 'til the sun rose Strange words come on out Of a grown man's mouth when his mind's broke Pictures and passin' time You only smile like that when you're drinkin' I wish I didn't, but I do Remember every moment on the nights with you "
    )

    prompt = (
        f"You are a powerful language model. Fill in the blanks in the following text with appropriate words. "
        f"The text is a part of a song with certain words masked by [MASK]. For each blank, think step by step \
        about the context and meaning of the surrounding text before choosing the word.\n\n"
        f"To do this, follow these steps:\n"
        f"  a. Carefully read and analysis the lyrics.\n"
        f"  b-1. Check the entire lyrics to see if there are any repeating parts.\n"
        f"  b-2. If repeating parts exist, replace the [MASK] with the corresponding word.\n"
        f"  c-1. Make the list of possible words for the masked part.\n"
        f"  c-2. Select a suitable word from the candidate list.\n"
        f"  c-3. Replace [MASK] with the word that you selected.\n"
        f"Example:\n\n"
        f"Lyrics: '{example_lyrics}'\n\n"
        f"Filled lyrics: '{example_filled}'\n\n"
        f"Now, based on the provided lyrics, fill in the blanks with appropriate words.\n\n"
        f"Lyrics: '{lyrics}'\n\n"
        f"Step-by-step reasoning and filled lyrics as 'Filled lyrics: {{the output}}'.\n\n"
        f"Say nothing but the filled lyrics as 'Filled lyrics: {{the output}}'.\n\n"
        f"Output example: Filled lyrics: 'I know this pain (I know this pain) why do you lock yourself up in these chains? (these chains)...'"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.0
    }
    
    return payload

def prepare_cot_few_shot_payload_KOR(lyrics):
    example_lyrics = (
        "세상에 음악의 신이 있다면 고맙다고 안아주고 싶어 전 세계 공통의 Language 자음과 모음이 달라도 상관없는 건 Music\
        말이 안 통해도 [MASK] 있다면 [MASK] 지금부터는 아주 친한 친구 너와 내가 모르는 사이여도 춤출 [MASK] 있어 We [MASK] mix it up right\
        Sugar and spice Brass sound and guitar 네 [MASK] 다 내 [MASK] 쿵치팍치 또한 내 이름인가\
        이것 또한 나를 위한 소린가 [MASK] [MASK] Drum bass Piano [MASK]\
        무엇이 우리의 행복인가 뭐 있나 춤을 춰 노래하자 [MASK] 우리의 행복이다\
        [MASK] 한번 더 Hey 음악은 우리의 숨이니까 위험하지 않아 계속 들이키자 Everybody 귀를 기울여 보자\
        세상에 음악의 신이 있다면 이건 당신께 주는 메시지 음정 하나하나 모아보자 음- 춤춰 노래해 기분이 끝내주네\
        쿵 치 팍 [MASK] 쿵 쿵 치 팍 [MASK] 예 쿵 치 팍 치 [MASK] [MASK] 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 [MASK] 예 행복은 바로 지금이야\
        생각해 봐 우리는 소음마저 음악이야 저마다의 [MASK] 맞춰가며 살아가 개미의 발소리마저도 Harmony\
        [MASK] [MASK] 행복인가 뭐 있나 춤을 춰 노래하자 이것이 우리의 행복이다 다시 한번 더 [MASK]\
        음악은 우리의 숨이니까 [MASK] 않아 계속 들이키자 Everybody 귀를 기울여 보자\
        세상에 [MASK] 신이 있다면 이건 당신께 주는 메시지 음정 하나하나 모아보자 음- 춤춰 노래해 기분이 끝내주네\
        [MASK] 치 팍 치 쿵 [MASK] 치 팍 치 예 [MASK] 치 팍 치 [MASK] 쿵 치 팍 치 예 [MASK] 치 팍 치 쿵 쿵 치 팍 치 예 음악의 신이 있다면\
        SEVENTEEN to the world 전 세계 다 합창 쿵 치 [MASK] 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 [MASK] 쿵 치 [MASK] 치 예\
        [MASK] 치 팍 치 쿵 [MASK] 치 팍 치 [MASK] 음악의 신이 있다면"
    )
    example_filled = (
        "세상에 음악의 신이 있다면 고맙다고 안아주고 싶어  전 세계 공통의 Language 자음과 모음이 달라도 상관없는 건 Music\
        말이 안 통해도 음악이 있다면 우리는 지금부터는 아주 친한 친구  너와 내가 모르는 사이여도 춤출 수 있어 We can mix it up right\
        Sugar and spice Brass sound and guitar  네 글자면 다 내 이름이래 쿵치팍치 또한 내 이름인가\
        이것 또한 나를 위한 소린가 Kick snare Drum bass Piano Bassline\
        무엇이 우리의 행복인가 뭐 있나 춤을 춰 노래하자 이것이 우리의 행복이다 \
        다시 한번 더 Hey  음악은 우리의 숨이니까 위험하지 않아 계속 들이키자 Everybody 귀를 기울여 보자\
        세상에 음악의 신이 있다면 이건 당신께 주는 메시지 음정 하나하나 모아보자 음- 춤춰 노래해 기분이 끝내주네\
        쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예 행복은 바로 지금이야\
        생각해 봐 우리는 소음마저 음악이야 저마다의 쿵짝 맞춰가며 살아가 개미의 발소리마저도 Harmony \
        무엇이 우리의 행복인가 뭐 있나 춤을 춰 노래하자 이것이 우리의 행복이다  다시 한번 더 Hey\
        음악은 우리의 숨이니까 위험하지 않아 계속 들이키자 Everybody 귀를 기울여 보자 \
        세상에 음악의 신이 있다면 이건 당신께 주는 메시지 음정 하나하나 모아보자 음- 춤춰 노래해 기분이 끝내주네 \
        쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예 음악의 신이 있다면 \
        SEVENTEEN to the world 전 세계 다 합창  쿵 치 팍 치 쿵 쿵 치 팍 치 예 쿵 치 팍 치 쿵 쿵 치 팍 치 예\
        쿵 치 팍 치 쿵 쿵 치 팍 치 예 음악의 신이 있다면"
    )

    prompt = (
        f"You are a powerful language model. Fill in the blanks in the following text with appropriate words. "
        f"The text is a part of a song with certain words masked by [MASK]. For each blank, think step by step \
        about the context and meaning of the surrounding text before choosing the word.\n\n"
        f"To do this, follow these steps:\n"
        f"  a. Carefully read and analysis the lyrics.\n"
        f"  b-1. Check the entire lyrics to see if there are any repeating parts.\n"
        f"  b-2. If repeating parts exist, replace the [MASK] with the corresponding word.\n"
        f"  c-1. Make the list of possible words for the masked part.\n"
        f"  c-2. Select a suitable word from the candidate list.\n"
        f"  c-3. Replace [MASK] with the word that you selected.\n"
        f"Example:\n\n"
        f"Lyrics: '{example_lyrics}'\n\n"
        f"Filled lyrics: '{example_filled}'\n\n"
        f"Now, based on the provided lyrics, fill in the blanks with appropriate words.\n\n"
        f"Lyrics: '{lyrics}'\n\n"
        f"Step-by-step reasoning and filled lyrics as 'Filled lyrics: {{the output}}'.\n\n"
        f"Say nothing but the filled lyrics as 'Filled lyrics: {{the output}}'.\n\n"
        f"Output example: Filled lyrics: 'I know this pain (I know this pain) why do you lock yourself up in these chains? (these chains)...'"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.0
    }
    
    return payload

# functions for reading file
def read_list_for_gpt(file_dir):
    with open(f'{file_dir}/list_for_gpt.txt', 'r', encoding='utf-8') as f:
        list_for_gpt = f.readlines()
    for list_idx in range(len(list_for_gpt)):
        list_for_gpt[list_idx] = list_for_gpt[list_idx].replace('\n', '')
    return list_for_gpt

def read_lyrics(file_dir, song, file_name='masked.txt'):
    dir = f'{file_dir}/{song}/{file_name}'
    with open(dir, 'r', encoding='utf-8') as f:
        lyrics = f.read()
    return lyrics.replace('\n', ' ')

# without batch API
def get_response_kor(openai_key, file_dir: str = './final_dataset/Korean_masking_task_words'):
    api_key = openai_key
    # read csv file
    # KOR

    song_list = read_list_for_gpt(file_dir)

    results = {
        'Title': [],
        'Original': [],
        'Masked': [],
        'Zero-shot': [],
        'COT': [],
        'COT_few-shot': []
    }

    for song in song_list:
        masked_text = read_lyrics(file_dir, song)
        original_text = read_lyrics(file_dir, song, 'original.txt')

        zero_shot_payload = prepare_zero_shot_payload(masked_text)
        zero_shot_response = make_request(zero_shot_payload, api_key)
        zero_shot_flag = (zero_shot_response['choices'][0]['finish_reason'] == 'stop')
        if zero_shot_flag:
            zero_shot_output = zero_shot_response['choices'][0]['message']['content']

        cot_payload = prepare_cot_payload(masked_text)
        cot_response = make_request(cot_payload, api_key)
        cot_flag = (cot_response['choices'][0]['finish_reason'] == 'stop')
        if cot_flag:
            cot_output = cot_response['choices'][0]['message']['content']

        cot_few_shot_payload = prepare_cot_few_shot_payload_KOR(masked_text)
        cot_few_shot_response = make_request(cot_few_shot_payload, api_key)
        cot_few_shot_flag = (cot_few_shot_response['choices'][0]['finish_reason'] == 'stop')
        if cot_few_shot_flag:
            cot_few_shot_output = cot_few_shot_response['choices'][0]['message']['content']

        if zero_shot_flag and cot_flag and cot_few_shot_flag:
            print(song)
            results['Title'].append(song)
            results['Original'].append(original_text)
            results['Masked'].append(masked_text)
            results['Zero-shot'].append(zero_shot_output)
            results['COT'].append(cot_output)
            results['COT_few-shot'].append(cot_few_shot_output)
        else:
            print(f'Error: {song}')
    
    df = pd.DataFrame(results)
    df.to_csv(f'{file_dir}/result.csv', index=False)

def get_response_eng(openai_key, file_dir: str = './final_dataset/English_masking_task_words'):
    api_key = openai_key

    song_list = read_list_for_gpt(file_dir)

    results = {
        'Title': [],
        'Original': [],
        'Masked': [],
        'Zero-shot': [],
        'COT': [],
        'COT_few-shot': []
    }

    for song in song_list:
        print(song)
        masked_text = read_lyrics(file_dir, song)
        original_text = read_lyrics(file_dir, song, 'original.txt')

        zero_shot_payload = prepare_zero_shot_payload(masked_text)
        zero_shot_response = make_request(zero_shot_payload, api_key)
        zero_shot_flag = (zero_shot_response['choices'][0]['finish_reason'] == 'stop')
        if zero_shot_flag:
            zero_shot_output = zero_shot_response['choices'][0]['message']['content']

        cot_payload = prepare_cot_payload(masked_text)
        cot_response = make_request(cot_payload, api_key)
        cot_flag = (cot_response['choices'][0]['finish_reason'] == 'stop')
        if cot_flag:
            cot_output = cot_response['choices'][0]['message']['content']

        cot_few_shot_payload = prepare_cot_few_shot_payload(masked_text)
        cot_few_shot_response = make_request(cot_few_shot_payload, api_key)
        cot_few_shot_flag = (cot_few_shot_response['choices'][0]['finish_reason'] == 'stop')
        if cot_few_shot_flag:
            cot_few_shot_output = cot_few_shot_response['choices'][0]['message']['content']

        # if zero_shot_flag and cot_flag and cot_few_shot_flag:
        if cot_few_shot_flag:
            results['Title'].append(song)
            results['Original'].append(original_text)
            results['Masked'].append(masked_text)
            results['Zero-shot'].append(zero_shot_output)
            results['COT'].append(cot_output)
            results['COT_few-shot'].append(cot_few_shot_output)

    
    df = pd.DataFrame(results)
    df.to_csv(f'{file_dir}/result.csv', index=False)

if __name__ == '__main__':
    api_key = "Enter your own API code"
