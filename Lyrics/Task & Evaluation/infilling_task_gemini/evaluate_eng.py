import csv
import tqdm

from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertModel
from rouge_score import rouge_scorer
from bert_score import BERTScorer

def read_csv(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data 

def calculate_rouge_eng(original, generated):
    # calculate the Rouge score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original, generated)
    avg_scores = [scores['rouge1'].recall, scores['rougeL'].recall]
    return avg_scores

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

def matching_answer_and_result(data_list, answer_data):
    """
        Match the answer with the generated result
    """
    new_data_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        for j in range(len(answer_data)):
            if data_list[i]['Title'] == answer_data[j]['Title']:
                data['Answer'] = answer_data[j]['Answer']
                new_data_list.append(data)
                break
    
    print(new_data_list)

    return new_data_list

def preprocess():
    file_name = "Gemini_infilling_result_ENG.csv"
    data_list = read_csv(file_name)
    answer_data = read_csv("ENG_lyrics_Evaluation_dataset.csv")

    data = matching_answer_and_result(data_list, answer_data)

    # write the result to a new csv file
    with open('Gemini_infilling_result.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Title', 'Original', 'Answer', 'Zero-shot', 'COT', 'COT_few_shot'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def evaluate_eng():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bertScorer = BERTScorer(model_type='bert-base-uncased')

    # read the respones from the csv file
    file_name = "Gemini_infilling_result.csv"
    data_list = read_csv(file_name)

    # calculate the Rouge score and BERT score
    scores = {'Rouge_zero-shot': 0., 'Rouge_COT': 0., 'Rouge_COT_few-shot': 0., 
            'Rouge_L_zero-shot': 0., 'Rouge_L_COT': 0., 'Rouge_L_COT_few-shot': 0.,
            'BERT_P_zero-shot': 0., 'BERT_P_COT': 0., 'BERT_P_COT_few-shot': 0.,
            'BERT_R_zero-shot': 0., 'BERT_R_COT': 0., 'BERT_R_COT_few-shot': 0.,
            'BERT_F1_zero-shot': 0., 'BERT_F1_COT': 0., 'BERT_F1_COT_few-shot': 0.
            }
    scores_list = []
    for data in tqdm.tqdm(data_list):
        # rouge score
        rouge_score_zero_shot = calculate_rouge_eng(data['Answer'], data['Zero-shot'].replace('Filled lyrics: ', '').replace('**', '').replace('\n', ' ').replace('  ', ' '))
        rouge_score_cot = calculate_rouge_eng(data['Answer'], data['COT'].replace('Filled lyrics: ', '').replace('**', '').replace('\n', ' ').replace('  ', ' '))
        rouge_score_cot_few_shot = calculate_rouge_eng(data['Answer'], data['COT_few_shot'].replace('Filled lyrics: ', '').replace('**', '').replace('\n', ' ').replace('  ', ' '))

        # bert score
        bert_score_zero_shot = calculate_bert_score(data['Answer'], data['Zero-shot'].replace('Filled lyrics: ', '').replace('**', '').replace('\n', ' ').replace('  ', ' '), bertScorer)
        bert_score_cot = calculate_bert_score(data['Answer'], data['COT'].replace('Filled lyrics: ', '').replace('**', '').replace('\n', ' ').replace('  ', ' '), bertScorer)
        bert_score_cot_few_shot = calculate_bert_score(data['Answer'], data['COT_few_shot'].replace('Filled lyrics: ', '').replace('**', '').replace('\n', ' ').replace('  ', ' '), bertScorer)

        scores_list.append({'Title': data['Title'], 'Rouge_zero-shot': round(rouge_score_zero_shot[0], 4), 'Rouge_COT': round(rouge_score_cot[0], 4), 'Rouge_COT_few-shot': round(rouge_score_cot_few_shot[0], 4), 
                    'Rouge_L_zero-shot': round(rouge_score_zero_shot[1], 4), 'Rouge_L_COT': round(rouge_score_cot[1], 4), 'Rouge_L_COT_few-shot': round(rouge_score_cot_few_shot[1], 4),
                    'BERT_P_zero-shot': round(bert_score_zero_shot[0], 4), 'BERT_P_COT': round(bert_score_cot[0], 4), 'BERT_P_COT_few-shot': round(bert_score_cot_few_shot[0], 4),
                    'BERT_R_zero-shot': round(bert_score_zero_shot[1], 4), 'BERT_R_COT': round(bert_score_cot[1], 4), 'BERT_R_COT_few-shot': round(bert_score_cot_few_shot[1], 4),
                    'BERT_F1_zero-shot': round(bert_score_zero_shot[2], 4), 'BERT_F1_COT': round(bert_score_cot[2], 4), 'BERT_F1_COT_few-shot': round(bert_score_cot_few_shot[2], 4)
                    })

        # get the average scores without saving each score
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
    with open('Gemini_infilling_result_scores.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in scores_list:
            writer.writerow(row)

if __name__ == '__main__':
    pass