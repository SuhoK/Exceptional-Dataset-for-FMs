import csv
import tqdm

from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertModel
from korouge_score import rouge_scorer
from KoBERTScore.KoBERTScore.score import bert_score

def read_csv(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data 

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

def calculate_bert_score_kor(tokenizer, model, original, generated):
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
    # bert_score_masked = score_from_all_layers(tokenizer, model, [row['Original']], [row['Masked']])

    # R, P, F1 = score_from_all_layers(tokenizer, model, chunks_original, chunks_generated)
    bert_scores_P = []
    bert_scores_R = []
    bert_scores_F1 = []
    for i in range(5):
        R, P, F1 = bert_score(tokenizer, model, [chunks_original[i]], [chunks_generated[i]])
    bert_scores_P.append(P)
    bert_scores_R.append(R)
    bert_scores_F1.append(F1)
    
    # return the average bert score
    return [(sum(bert_scores_P) / len(bert_scores_P)).tolist()[0], (sum(bert_scores_R) / len(bert_scores_R)).tolist()[0], (sum(bert_scores_F1) / len(bert_scores_F1)).tolist()[0]]

def evaluate_kor():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    data_list = read_csv('Gemini_infilling_result_KOR.csv')
    scores_list = []
    scores = {'Rouge_zero-shot': 0., 'Rouge_COT': 0., 'Rouge_COT_few-shot': 0., 
            'Rouge_L_zero-shot': 0., 'Rouge_L_COT': 0., 'Rouge_L_COT_few-shot': 0.,
            'BERT_P_zero-shot': 0., 'BERT_P_COT': 0., 'BERT_P_COT_few-shot': 0.,
            'BERT_R_zero-shot': 0., 'BERT_R_COT': 0., 'BERT_R_COT_few-shot': 0.,
            'BERT_F1_zero-shot': 0., 'BERT_F1_COT': 0., 'BERT_F1_COT_few-shot': 0.
            }

    for data in tqdm.tqdm(data_list):
        # calculate the Rouge score
        rouge_score_zero_shot = calculate_rouge_ko(data['Original'], data['Zero-shot'])
        rouge_score_cot = calculate_rouge_ko(data['Original'], data['COT'])
        rouge_score_cot_few_shot = calculate_rouge_ko(data['Original'], data['COT_few_shot'])

        # calculate the BERT score
        bert_score_zero_shot = calculate_bert_score_kor(tokenizer, model, data['Original'], data['Zero-shot'])
        bert_score_cot = calculate_bert_score_kor(tokenizer, model, data['Original'], data['COT'])
        bert_score_cot_few_shot = calculate_bert_score_kor(tokenizer, model, data['Original'], data['COT_few_shot'])

        scores_list.append({'Title': data['Title'], 'Rouge_zero-shot': round(rouge_score_zero_shot[0], 4), 'Rouge_COT': round(rouge_score_cot[0], 4), 'Rouge_COT_few-shot': round(rouge_score_cot_few_shot[0], 4), 
                    'Rouge_L_zero-shot': round(rouge_score_zero_shot[1], 4), 'Rouge_L_COT': round(rouge_score_cot[1], 4), 'Rouge_L_COT_few-shot': round(rouge_score_cot_few_shot[1], 4),
                    'BERT_P_zero-shot': round(bert_score_zero_shot[0], 4), 'BERT_P_COT': round(bert_score_cot[0], 4), 'BERT_P_COT_few-shot': round(bert_score_cot_few_shot[0], 4),
                    'BERT_R_zero-shot': round(bert_score_zero_shot[1], 4), 'BERT_R_COT': round(bert_score_cot[1], 4), 'BERT_R_COT_few-shot': round(bert_score_cot_few_shot[1], 4),
                    'BERT_F1_zero-shot': round(bert_score_zero_shot[2], 4), 'BERT_F1_COT': round(bert_score_cot[2], 4), 'BERT_F1_COT_few-shot': round(bert_score_cot_few_shot[2], 4)
                   })
        
        # add the scores to the data
        scores['Rouge_zero-shot'] += rouge_score_zero_shot[0]
        scores['Rouge_COT'] += rouge_score_cot[0]
        scores['Rouge_COT_few-shot'] += rouge_score_cot_few_shot[0]

        scores['Rouge_L_zero-shot'] += rouge_score_zero_shot[1]
        scores['Rouge_L_COT'] += rouge_score_cot[1]
        scores['Rouge_L_COT_few-shot'] += rouge_score_cot_few_shot[1]

        scores['BERT_P_zero-shot'] += bert_score_zero_shot[0]
        scores['BERT_P_COT'] += bert_score_cot[0]
        scores['BERT_P_COT_few-shot'] += bert_score_cot_few_shot[0]

        scores['BERT_R_zero-shot'] += bert_score_zero_shot[1]
        scores['BERT_R_COT'] += bert_score_cot[1]
        scores['BERT_R_COT_few-shot'] += bert_score_cot_few_shot[1]

        scores['BERT_F1_zero-shot'] += bert_score_zero_shot[2]
        scores['BERT_F1_COT'] += bert_score_cot[2]
        scores['BERT_F1_COT_few-shot'] += bert_score_cot_few_shot[2]

    # print the average scores
    num_data = len(data_list)
    print(f'Rouge score: {round(scores["Rouge_zero-shot"]/num_data, 4)}, {round(scores["Rouge_COT"]/num_data, 4)}, {round(scores["Rouge_COT_few-shot"]/num_data, 4)}')
    print(f'Rouge_L score: {round(scores["Rouge_L_zero-shot"]/num_data, 4)}, {round(scores["Rouge_L_COT"]/num_data, 4)}, {round(scores["Rouge_L_COT_few-shot"]/num_data, 4)}')
    print(f'BERT_P score: {round(scores["BERT_P_zero-shot"]/num_data, 4)}, {round(scores["BERT_P_COT"]/num_data, 4)}, {round(scores["BERT_P_COT_few-shot"]/num_data, 4)}')
    print(f'BERT_R score: {round(scores["BERT_R_zero-shot"]/num_data, 4)}, {round(scores["BERT_R_COT"]/num_data, 4)}, {round(scores["BERT_R_COT_few-shot"]/num_data, 4)}')
    print(f'BERT_F1 score: {round(scores["BERT_F1_zero-shot"]/num_data, 4)}, {round(scores["BERT_F1_COT"]/num_data, 4)}, {round(scores["BERT_F1_COT_few-shot"]/num_data, 4)}')

    fieldnames = ['Title', 'Rouge_zero-shot', 'Rouge_COT', 'Rouge_COT_few-shot', 
                  'Rouge_L_zero-shot', 'Rouge_L_COT', 'Rouge_L_COT_few-shot', 
                  'BERT_P_zero-shot', 'BERT_P_COT', 'BERT_P_COT_few-shot', 
                  'BERT_R_zero-shot', 'BERT_R_COT', 'BERT_R_COT_few-shot', 
                  'BERT_F1_zero-shot', 'BERT_F1_COT', 'BERT_F1_COT_few-shot'
                  ]

    with open('Gemini_infilling_result_scores_KOR.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in scores_list:
            writer.writerow(row)

if __name__ == "__main__":
    pass