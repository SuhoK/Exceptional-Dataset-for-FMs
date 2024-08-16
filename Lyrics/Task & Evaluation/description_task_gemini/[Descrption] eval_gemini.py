import os
import warnings
import pandas as pd
import csv
import tqdm
from rouge_score import rouge_scorer
from bert_score import BERTScorer


# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")

def evaluate_using_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    avg_scores = {
        'rouge-1': scores['rouge1'],
        'rouge-l': scores['rougeL']
    }
    return avg_scores

def evaluate_using_bert_score(text, predicted_text, scorer):
    P, R, F1 = scorer.score([text], [predicted_text])
    return P.mean().item(), R.mean().item(), F1.mean().item()  # Calculate mean scores

def read_csv(path = 'description_result_#3.csv'):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# Initializing BERT scorer
scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# Load the ground truth descriptions
ground_truth_df = read_csv('Billboard_weekly_filtered.csv')

# Read result
result_files = ['description_result_#3.csv']

# Loop through the sorted CSV files
for csv_file in result_files:
    df = read_csv(csv_file)
    
    n = len(df)
    total_scores = {
        'Zero-shot': {'rouge1_f': 0, 'rouge1_r': 0, 'rouge1_p': 0, 'rougeL_f': 0, 'rougeL_r': 0, 'rougeL_p': 0, 'bert_p': 0, 'bert_r': 0, 'bert_f1': 0},
        'COT': {'rouge1_f': 0, 'rouge1_r': 0, 'rouge1_p': 0, 'rougeL_f': 0, 'rougeL_r': 0, 'rougeL_p': 0, 'bert_p': 0, 'bert_r': 0, 'bert_f1': 0},
        'COT_few_shot': {'rouge1_f': 0, 'rouge1_r': 0, 'rouge1_p': 0, 'rougeL_f': 0, 'rougeL_r': 0, 'rougeL_p': 0, 'bert_p': 0, 'bert_r': 0, 'bert_f1': 0}
    }

    # Iterate through the merged dataframe and calculate scores for each description type
    for row in tqdm.tqdm(df):
        reference = row['Description']
        
        for desc_type in ['Zero-shot', 'COT', 'COT_few_shot']:
            hypothesis = row[f'{desc_type}']
            
            # Evaluate using ROUGE
            rouge_scores = evaluate_using_rouge(reference, hypothesis)
            total_scores[desc_type]['rouge1_f'] += rouge_scores['rouge-1'].fmeasure
            total_scores[desc_type]['rouge1_r'] += rouge_scores['rouge-1'].recall
            total_scores[desc_type]['rouge1_p'] += rouge_scores['rouge-1'].precision
            total_scores[desc_type]['rougeL_f'] += rouge_scores['rouge-l'].fmeasure
            total_scores[desc_type]['rougeL_r'] += rouge_scores['rouge-l'].recall
            total_scores[desc_type]['rougeL_p'] += rouge_scores['rouge-l'].precision
            
            # Evaluate using BERTScore
            P, R, F1 = evaluate_using_bert_score(reference, hypothesis, scorer)
            total_scores[desc_type]['bert_p'] += P
            total_scores[desc_type]['bert_r'] += R
            total_scores[desc_type]['bert_f1'] += F1

    # Calculate the average scores for this file
    if n > 0:
        avg_scores = {
            desc_type: {
                'avg_rouge1_f': round(total_scores[desc_type]['rouge1_f'] / n, 4),
                'avg_rouge1_r': round(total_scores[desc_type]['rouge1_r'] / n, 4),
                'avg_rouge1_p': round(total_scores[desc_type]['rouge1_p'] / n, 4),
                'avg_rougeL_f': round(total_scores[desc_type]['rougeL_f'] / n, 4),
                'avg_rougeL_r': round(total_scores[desc_type]['rougeL_r'] / n, 4),
                'avg_rougeL_p': round(total_scores[desc_type]['rougeL_p'] / n, 4),
                'avg_bert_p': round(total_scores[desc_type]['bert_p'] / n, 4),
                'avg_bert_r': round(total_scores[desc_type]['bert_r'] / n, 4),
                'avg_bert_f1': round(total_scores[desc_type]['bert_f1'] / n, 4)
            }
            for desc_type in ['Zero-shot', 'COT', 'COT_few_shot']
        }

print(avg_scores)