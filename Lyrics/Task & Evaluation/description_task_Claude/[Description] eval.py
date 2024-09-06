import os
import warnings
import pandas as pd
import csv
import tqdm
from rouge_score import rouge_scorer
from bert_score import BERTScorer


result_1 = pd.read_csv(os.path.join(ROOT_DIR, "Claude_description_weekly_result.csv"), header=0)
result_2 = pd.read_csv(os.path.join(ROOT_DIR, "Claude_description_weekly_result_#2.csv"), header=0)
merged_df = pd.merge(result_1, result_2, how="outer")


# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")

def evaluate_using_rouge(reference, hypothesis):
    """
    Evaluate ROUGE scores between reference and hypothesis.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    avg_scores = {
        'rouge-1': scores['rouge1'],
        'rouge-l': scores['rougeL']
    }
    return avg_scores

def evaluate_using_bert_score(text, predicted_text, scorer):
    """
    Evaluate BERTScore between text and predicted_text.
    """
    P, R, F1 = scorer.score([text], [predicted_text])
    return P.mean().item(), R.mean().item(), F1.mean().item()

def read_csv(path):
    """
    Read CSV file and return data as a list of dictionaries.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def clean_description(description):
    """
    Clean and extract the description from the given text.
    """
    if not isinstance(description, str):
        return ""  # Handle non-string inputs

    start = description.find("Description: ") + len("Description: ")
    end = description.rfind(" }")
    if start == -1 or end == -1:
        return description  # Return as is if delimiters not found
    return description[start:end].strip()

# Initialize BERT scorer
scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# Example DataFrame definition (replace with actual DataFrame)
# merged_df = pd.read_csv('your_file.csv')  # Ensure this is defined

# Evaluation
df = merged_df.fillna('')
n = len(df)
total_scores = {
    'Zero-shot': {'rouge1_f': 0, 'rouge1_r': 0, 'rouge1_p': 0, 'rougeL_f': 0, 'rougeL_r': 0, 'rougeL_p': 0, 'bert_p': 0, 'bert_r': 0, 'bert_f1': 0},
    'COT': {'rouge1_f': 0, 'rouge1_r': 0, 'rouge1_p': 0, 'rougeL_f': 0, 'rougeL_r': 0, 'rougeL_p': 0, 'bert_p': 0, 'bert_r': 0, 'bert_f1': 0},
    'COT_few_shot': {'rouge1_f': 0, 'rouge1_r': 0, 'rouge1_p': 0, 'rougeL_f': 0, 'rougeL_r': 0, 'rougeL_p': 0, 'bert_p': 0, 'bert_r': 0, 'bert_f1': 0}
}

# Iterate through the merged DataFrame and calculate scores
for row in tqdm.tqdm(df.iterrows(), total=n):
    index, row = row
    reference = clean_description(row['description'])
    
    for desc_type in ['Zero-shot', 'COT', 'COT_few_shot']:
        hypothesis = clean_description(row[desc_type])
        
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

# Calculate the average scores
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
