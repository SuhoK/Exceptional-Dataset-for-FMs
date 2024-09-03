import torch
import os
from tqdm import tqdm
from bert_score import BERTScorer
from rouge import Rouge
from transformers import BertTokenizer, BertForMaskedLM

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
    text = ''
    with open(file_name, 'r') as f:
        for line in f:
            text += line + ' '

    return text

def predicted_text(outputs, tokenizer):
    predicted_text = ''
    logits = outputs.logits
    softmax = torch.nn.Softmax(dim=-1)
    for i in range(logits.size(1)):
        mask_word_logits = softmax(logits[0, i])
        predicted_index = torch.argmax(mask_word_logits).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        predicted_text += (tokenizer.convert_tokens_to_string(predicted_token) + ' ')
    
    return predicted_text

def predict_with_BERT(text, masked_text, tokenizer, model):
    # Using BERT, evaluate the performance of the model on the test set
    
    origin = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs = tokenizer(masked_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    
    inputs['labels'] = origin['input_ids']

    with torch.no_grad():
        outputs = model(**inputs)

    return origin, outputs

def evaluate_using_rouge(text, predicted_text):
    # Evaluate the performance of the model using ROUGE
    rouge = Rouge()
    model_score = rouge.get_scores(text, predicted_text)
    return model_score

def evaluate_using_bert_score(text, predicted_text, scorer):
    # Evaluate the performance of the model using BERTScore
    P, R, F1 = scorer.score([text], [predicted_text])
    return P, R, F1

def write_file(file_name, text):
    # Write the text to the file
    with open(file_name, 'w') as f:
        f.write(text)

def evaluate_all(dir_name, tokenizer, model, bertScorer):
    masked_url = f'{dir_name}/masked.txt'
    masked_text = read_file(masked_url)
    
    clean_url = f'{dir_name}/original.txt'
    text = read_file(clean_url)

    inputs, outputs = predict_with_BERT(text, masked_text, tokenizer, model)
    model_predicted_text = predicted_text(outputs, tokenizer)

    inputs_to_text = ''.join(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][1:-1])))

    # Get scores
    ROUGE_score = evaluate_using_rouge(inputs_to_text, model_predicted_text)
    BERT_score = evaluate_using_bert_score(inputs_to_text, model_predicted_text, bertScorer)

    BERT_score_string = f'{BERT_score[2].item()}'

    write_file(f'{dir_name}/bert_predict_scores.txt', f'{ROUGE_score[0]['rouge-1']['r']}\n{BERT_score_string}')
    
    return [ROUGE_score[0]['rouge-1']['r'], BERT_score[2].item()]

def evaluate_bert_eng():
    file_names = get_file_names('./final_dataset/English_masking_task_token')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bertScorer = BERTScorer(model_type='bert-base-uncased')

    token_bert_sum = 0
    token_num = 0
    word_bert_sum = 0
    word_num = 0

    for i in tqdm(range(len(file_names))):
        file_name = file_names[i]
        # masking_task_words, masking_task_tokens
        dir_name_word = f'./final_dataset/English_masking_task_words/{file_name}'
        dir_name_token = f'./final_dataset/English_masking_task_token/{file_name}'
        if not get_file_names(dir_name_word) == []:
            tmp = evaluate_all(dir_name_word, tokenizer, model, bertScorer)
            # word_rouge_sum += tmp[0]
            word_bert_sum += tmp[1]
            word_num += 1

        if not get_file_names(dir_name_token) == []:
            tmp = evaluate_all(dir_name_token, tokenizer, model, bertScorer)
            # token_rouge_sum += tmp[0]
            token_bert_sum += tmp[1]
            token_num += 1
    
    # print(f'word rouge-1: {word_rouge_sum / word_num}')
    print(f'word bert: {word_bert_sum / word_num}')
    print(f'word num: {word_num}')

    # print(f'token rouge-1: {token_rouge_sum / token_num}')
    print(f'token bert: {token_bert_sum / token_num}')
    print(f'token num: {token_num}')

if __name__ == '__main__':
    evaluate_bert_eng()