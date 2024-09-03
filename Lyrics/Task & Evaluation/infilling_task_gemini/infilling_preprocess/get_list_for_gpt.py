import os

def get_file_names(directory):
    # Get all the name of files in the directory
    # if directory is not directory return nothing
    if not os.path.isdir(directory):
        return []
    file_names = os.listdir(directory)
    file_names.sort()
    return file_names

def read_file(file_name):
    # Get the text from the file
    text = []
    with open(file_name, 'r') as f:
        for line in f:
            text.append(line)

    return text

def get_score(task_dir, dir_name):
    score_list = read_file(f'{task_dir}/{dir_name}/bert_predict_scores.txt')
    score = 0
    if ':' in score_list[-1]:
        score = float(score_list[-1].split(': ')[1].strip())
    else:
        score = float(score_list[-1].strip())
    
    return score

def get_list_for_gpt(task_dir, target_score):
    dir_names = get_file_names(task_dir)

    count = 0
    list_for_gpt = []
    for dir_name in dir_names:
        file_names = get_file_names(f'{task_dir}/{dir_name}')
        if file_names != []:
            score = get_score(task_dir, dir_name)
            if score < target_score:
                list_for_gpt.append(dir_name)
            else:
                count += 1
    print(f'count: {count}')
    
    # sort list_for_gpt by dir_num
    list_for_gpt.sort(key=lambda x: int(x.split('_')[0]))

    text_lyrics = ''
    with open(f'{task_dir}/list_for_gpt.txt', 'w') as f:
        for dir_name in list_for_gpt:
            f.write(f'{dir_name}\n')
            text = read_file(f'{task_dir}/{dir_name}/masked.txt')
            text_lyrics += ''.join(text)
    
    with open(f'{task_dir}/list_for_gpt_lyrics.txt', 'w') as f:
        f.write(text_lyrics)


if __name__ == '__main__':
    task_dir = './final_dataset/English_masking_task_words'
    get_list_for_gpt(task_dir, 0.9)
