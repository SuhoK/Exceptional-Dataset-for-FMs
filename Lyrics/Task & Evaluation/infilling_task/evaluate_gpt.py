import os
import csv
import argparse
from bert_score import BERTScorer
from rouge import Rouge
from rouge_score import rouge_scorer as eng_rouge_scorer
from korouge_score import rouge_scorer
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from sklearn.metrics.pairwise import cosine_similarity

# read file
def read_csv(file_dir):
    data = []
    with open(file_dir, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def read_results(file_dir):
    print(file_dir)
    data_for_2024 = read_csv(file_dir)

    return data_for_2024

def read_lyrics(file_dir, song):
    dir = f'{file_dir}/{song}/original.txt'
    with open(dir, 'r', encoding='utf-8') as f:
        lyrics = f.read()
    return lyrics.replace('\n', ' ')

# Evaluation
def get_embeddings(text_list, tokenizer, model):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.hidden_states[-1].mean(dim=1).detach().numpy()
    return embeddings

def evaluate_using_cosine_similarity(inputs_to_text, model_predicted_text, model, tokenizer):
    input_list = inputs_to_text.split('\n')
    predicted_list = model_predicted_text.replace('\n\n', '\n').split('\n')
    input_embedding = get_embeddings(input_list, tokenizer, model)
    predicted_embedding = get_embeddings(predicted_list, tokenizer, model)

    scores = cosine_similarity(predicted_embedding, input_embedding)
    return scores

def calculate_rouge_eng(original, generated):
    # calculate the Rouge score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original, generated)
    avg_scores = [scores['rouge1'].recall, scores['rougeL'].recall]
    return avg_scores

def calculate_rouge_ko(original, generated):
    # calculate the Rouge score
    generated = generated.replace('Filled lyrics: ', '')
    generated_len = len(generated.split(' '))
    original_len = len(original.split(' '))
    num_chunks = min(generated_len, original_len) // 500 + 1
    gen_chunk_len = generated_len // num_chunks
    orig_chunk_len = original_len // num_chunks

    generated_chunks = [generated[i:i+gen_chunk_len] for i in range(0, len(generated), gen_chunk_len)]
    original_chunks = [original[i:i+orig_chunk_len] for i in range(0, len(original), orig_chunk_len)]

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    model_score_r = 0
    model_score_l = 0
    for i in range(num_chunks):
        model_score_r += rouge.score(original_chunks[i], generated_chunks[i])['rouge1'].recall
        model_score_l += rouge.score(original_chunks[i], generated_chunks[i])['rougeL'].recall
    return [model_score_r / num_chunks, model_score_l / num_chunks]

def calculate_bert_score(original, generated, scorer):
    """
        Calculate the BERT score
        First, chunk the data into 5 chunks
        Second, calculate the BERT score for each chunk
        Third, calculate the average BERT score for all chunks
    """
    # chunk the data into 5 chunks
    generated = generated.replace('Filled lyrics: ', '')

    chunk_size_original = len(original) // 5
    chunk_size_generated = len(generated) // 5

    chunks_original = [original[i:i+chunk_size_original] for i in range(0, len(original), chunk_size_original)]
    chunks_generated = [generated[i:i+chunk_size_generated] for i in range(0, len(generated), chunk_size_generated)]

    # calculate bert score for each chunk
    bert_scores_P = []
    bert_scores_R = []
    bert_scores_F1 = []
    for i in range(5):
        P, R, F1 = scorer.score([chunks_original[i]], [chunks_generated[i]])
    bert_scores_P.append(P)
    bert_scores_R.append(R)
    bert_scores_F1.append(F1)
    
    # return the average bert score
    return [(sum(bert_scores_P) / len(bert_scores_P)).tolist()[0], (sum(bert_scores_R) / len(bert_scores_R)).tolist()[0], (sum(bert_scores_F1) / len(bert_scores_F1)).tolist()[0]]

def evaluate_gpt_eng(dataset_dir: str = './final_dataset/English_masking_task_words', 
                     write_csv_file: str = './final_dataset/ENG_evaluation_results.csv'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bertScorer = BERTScorer(model_type='bert-base-uncased')

    file_dir = f'{dataset_dir}'
    results_of_2024 = read_results(file_dir)

    scores = []
    for row in results_of_2024:
        if not row['Zero-shot'].startswith("Filled lyrics:"):
            continue

        print(f'name: {row['Title']}')
        row['COT_few-shot'] = row['COT_few-shot'].replace("[", '').replace("]", '')
        # evaluate the results
        # calculate the Rouge score
        rouge_score_zero_shot = calculate_rouge_eng(row['Original'], row['Zero-shot'])
        rouge_score_cot = calculate_rouge_eng(row['Original'], row['COT'])
        rouge_score_cot_few_shot = calculate_rouge_eng(row['Original'], row['COT_few-shot'])

        # calculate BERT score
        # result is now tensor. Convert it to float
        bert_score_zero_shot = calculate_bert_score(row['Original'], row['Zero-shot'], bertScorer)
        bert_score_cot = calculate_bert_score(row['Original'], row['COT'], bertScorer)
        bert_score_cot_few_shot = calculate_bert_score(row['Original'], row['COT_few-shot'], bertScorer)

        scores.append({'Title': row['Title'], 'Rouge_zero-shot': round(rouge_score_zero_shot[0], 4), 'Rouge_COT': round(rouge_score_cot[0], 4), 'Rouge_COT_few-shot': round(rouge_score_cot_few_shot[0], 4), 
                    'Rouge_L_zero-shot': round(rouge_score_zero_shot[1], 4), 'Rouge_L_COT': round(rouge_score_cot[1], 4), 'Rouge_L_COT_few-shot': round(rouge_score_cot_few_shot[1], 4),
                    'BERT_P_zero-shot': round(bert_score_zero_shot[0], 4), 'BERT_P_COT': round(bert_score_cot[0], 4), 'BERT_P_COT_few-shot': round(bert_score_cot_few_shot[0], 4),
                    'BERT_R_zero-shot': round(bert_score_zero_shot[1], 4), 'BERT_R_COT': round(bert_score_cot[1], 4), 'BERT_R_COT_few-shot': round(bert_score_cot_few_shot[1], 4),
                    'BERT_F1_zero-shot': round(bert_score_zero_shot[2], 4), 'BERT_F1_COT': round(bert_score_cot[2], 4), 'BERT_F1_COT_few-shot': round(bert_score_cot_few_shot[2], 4)
                    })

    fieldnames = ['Title', 'Rouge_zero-shot', 'Rouge_COT', 'Rouge_COT_few-shot', 
                    'Rouge_L_zero-shot', 'Rouge_L_COT', 'Rouge_L_COT_few-shot', 
                    'BERT_P_zero-shot', 'BERT_P_COT', 'BERT_P_COT_few-shot', 
                    'BERT_R_zero-shot', 'BERT_R_COT', 'BERT_R_COT_few-shot', 
                    'BERT_F1_zero-shot', 'BERT_F1_COT', 'BERT_F1_COT_few-shot'
                    ]

    with open(write_csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scores:
            writer.writerow(row)

def evaluate_gpt_kor(dataset_dir: str = './final_dataset/Korean_masking_task_words',
                     write_csv_file: str = './final_dataset/KOR_evaluation_results.csv'):

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

    config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased', config=config)

    file_dir = f'{dataset_dir}'
    results_of_2024 = read_results(file_dir)

    print(file_dir)

    scores = []
    for row in results_of_2024:
        print(f'name: {row['Title']}')
        row['Original'] = read_lyrics(dataset_dir, row['Title'])

        # evaluate the results
        # calculate the Rouge score
        rouge_score_zero_shot = calculate_rouge_ko(row['Original'], row['Zero-shot'])
        rouge_score_cot = calculate_rouge_ko(row['Original'], row['COT'])
        rouge_score_cot_few_shot = calculate_rouge_ko(row['Original'], row['COT_few-shot'])

        # calculate cosine similarity
        scores_zero_shot = evaluate_using_cosine_similarity(row['Original'], row['Zero-shot'], model, tokenizer)
        scores_cot = evaluate_using_cosine_similarity(row['Original'], row['COT'], model, tokenizer)
        scores_cot_few_shot = evaluate_using_cosine_similarity(row['Original'], row['COT_few-shot'], model, tokenizer)

        scores.append({'Title': row['Title'], 'Rouge_zero-shot': round(rouge_score_zero_shot[0], 4), 'Rouge_COT': round(rouge_score_cot[0], 4), 'Rouge_COT_few-shot': round(rouge_score_cot_few_shot[0], 4), 
                    'Rouge_L_zero-shot': round(rouge_score_zero_shot[1], 4), 'Rouge_L_COT': round(rouge_score_cot[1], 4), 'Rouge_L_COT_few-shot': round(rouge_score_cot_few_shot[1], 4),
                    'Cosine_similarity_zero-shot': round(scores_zero_shot[0][0], 4), 'Cosine_similarity_COT': round(scores_cot[0][0], 4), 'Cosine_similarity_COT_few-shot': round(scores_cot_few_shot[0][0], 4)
                   })
    fieldnames = ['Title', 'Rouge_zero-shot', 'Rouge_COT', 'Rouge_COT_few-shot', 
                  'Rouge_L_zero-shot', 'Rouge_L_COT', 'Rouge_L_COT_few-shot', 
                  'Cosine_similarity_zero-shot', 'Cosine_similarity_COT', 'Cosine_similarity_COT_few-shot'
                  ]

    with open(write_csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scores:
            writer.writerow(row)


if __name__ == '__main__':
    pass
