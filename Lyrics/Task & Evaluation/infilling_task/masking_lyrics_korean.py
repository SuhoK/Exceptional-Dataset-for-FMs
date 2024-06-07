import os
import pandas as pd
import torch
from transformers import BertTokenizer

def divide_lyrics(lyrics):
    lyrics_by_line = lyrics.split('\n')
    lyrics_block = []
    i = 0
    for line in lyrics_by_line:
        if i%5 == 0:
            lyrics_block.append(line)
        else:
            lyrics_block[-1] += (' ' + line)
        i += 1
    return lyrics_block

def make_directory(directory):
    """
    Create a directory for the text files.
    """
    # Define the directory name
    directory = directory

    if os.path.exists(directory):
        # If the directory already exists, do nothing
        return

    # Create the directory
    os.makedirs(directory, exist_ok=True)

def masking_words(text):
    # split the text by space
    input = text.split(' ')
    # drop the element if it is empty
    input = [i for i in input if i]

    # count the number of words
    num_words = len(input)

    # randomly select the index of the word to be masked
    rand = torch.rand(num_words)
    mask_arr = (rand < 0.15)

    selection = torch.flatten(mask_arr.nonzero()).tolist()
    for i in selection:
        input[i] = '[MASK]'

    after_masking = ' '.join(input)

    return selection, after_masking.replace(' \n', '\n')

def mask_all():
    path_korean_songs = './final_dataset/melon_chart_combined_2024.csv'
    df = pd.read_csv(path_korean_songs)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    row_number = 0
    text = ''
    for index, row in df.iterrows():
        original = ''
        masked = ''
        lyrics = divide_lyrics(row['Lyrics'])

        for lyric in lyrics:
            selection, masked_text = masking_words(lyric)
            original += lyric + '\n'
            masked += masked_text + '\n'
        
        final_dir = f'./final_dataset/korean_masking_task_2024/{row_number}_{row["Title"]}'
        make_directory(final_dir)
        row_number += 1

        text += f'{row_number}_{row["Title"]}\n'

        # write at the file
        # original.txt
        # masked.txt
        with open(f'{final_dir}/original.txt', 'w') as f:
            f.write(original)
        with open(f'{final_dir}/masked.txt', 'w') as f:
            f.write(masked)

    with open('./final_dataset/korean_masking_task_2024/list_for_gpt.txt', 'w') as f:
        f.write(text)


if __name__ == '__main__':
    # korean_songs_all_with_genre_lyrics_description.csv file에서 Lyrics만 추출
    pass