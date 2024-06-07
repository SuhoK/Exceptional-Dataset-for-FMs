import os
import torch
from transformers import BertTokenizer, BertForMaskedLM

def get_file_names(directory):
    # Get all the name of files in the directory
    # if directory is not directory return nothing
    if not os.path.isdir(directory):
        return []
    file_names = os.listdir(directory)
    file_names.sort()
    return file_names

def get_texts(file_name):
    # Get the text from the file
    text = ''
    with open(file_name, 'r') as f:
        for line in f:
            if '\n' in line:
                line = line.replace('\n', '')
            text += line + ' '

    return text

def masking_tokens(text, tokenizer):
    inputs = tokenizer(text, return_tensors='pt')

    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)

    selection = torch.flatten(mask_arr[0].nonzero()).tolist()
    inputs.input_ids[0, selection] = 103

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][1:-1])
    after_masking = ''.join(tokenizer.convert_tokens_to_string(tokens))

    return selection, after_masking

def masking_words(text):
    # split the text by space
    input = text.split(' ')

    # count the number of words
    num_words = len(input)

    # randomly select the index of the word to be masked
    rand = torch.rand(num_words)
    mask_arr = (rand < 0.15)

    selection = torch.flatten(mask_arr.nonzero()).tolist()
    for i in selection:
        input[i] = '[MASK]'
    
    after_masking = ' '.join(input)

    return selection, after_masking

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

def chunk_text(text, max_len):
    # Split the text into chunks of max_len
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

def mask_all():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dir_path = f'./lyrics_only_txt'
    dir_names = get_file_names(dir_path)
    for dir_name in dir_names:
        # if int(dir_name.split('_')[0]) < 1924:
        #     continue
        dir = f'{dir_path}/{dir_name}'
        file_names = get_file_names(dir)
        
        # strings to store the original and masked text
        original = ''
        masked = ''
        masked_word = ''
        if file_names != []:
            for file_name in file_names:
                text = get_texts(f'{dir}/{file_name}')
                chunked_text = chunk_text(text, 1200)

                if len(chunked_text) > 1:
                    for chunk in chunked_text:
                        selection, masked_text = masking_tokens(chunk, tokenizer)
                        selection_word, masked_text_word = masking_words(chunk)

                        original += chunk + '\n'
                        masked += masked_text + '\n'
                        masked_word += masked_text_word + '\n'
                else:
                    selection, masked_text = masking_tokens(text, tokenizer)
                    selection_word, masked_text_word = masking_words(text)

                    original += text + '\n'
                    masked += masked_text + '\n'
                    masked_word += masked_text_word + '\n'
            
            final_dir_token = f'./final_dataset/English_masking_task_token/{dir_name}'
            make_directory(final_dir_token)
            # save original and masked in directory as separate text files
            with open(f'{final_dir_token}/original.txt', 'w') as f:
                f.write(original)
            with open(f'{final_dir_token}/masked.txt', 'w') as f:
                f.write(masked)
            
            final_dir_word = f'./final_dataset/English_masking_task_words/{dir_name}'
            make_directory(final_dir_word)
            # save original and masked in directory as separate text files
            with open(f'{final_dir_word}/original.txt', 'w') as f:
                f.write(original)
            with open(f'{final_dir_word}/masked.txt', 'w') as f:
                f.write(masked_word)

if __name__ == '__main__':
    pass