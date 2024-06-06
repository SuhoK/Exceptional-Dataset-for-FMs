!pip install pandas torchmetrics
from torchmetrics.text import WordErrorRate
import pandas as pd
import re

def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text
  
def precision(pred_li, gt_li):
    cor, tol = 0, 0
    for gt, pred in zip(gt_li, pred_li):
        for i in pred:
            if i in gt:
                cor += 1
        tol += len(pred)
    return cor / tol * 100

def recall(pred_li, gt_li):
    cor, tol = 0, 0
    for gt, pred in zip(gt_li, pred_li):
        for i in gt:
            if i in pred:
                cor += 1
        tol += len(gt)
    return cor / tol * 100

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)
  
def accuracy(pred, target):
    correct = sum(p == t for p, t in zip(pred, target))
    total = len(target)
    return correct / total if total > 0 else 0


def calculate_wer_from_csv(preds_csv_path, gt_csv_path):
    preds_df = pd.read_csv(preds_csv_path)
    target_df = pd.read_csv(gt_csv_path)

    merged_df = pd.merge(preds_df, target_df, on='filename', suffixes=('_pred', '_target'))
    preds = merged_df[preds_column + '_pred'].tolist()
    gt = merged_df[target_column + '_gt'].tolist()

    wer = WordErrorRate()
    wer_score = wer(preds, gt).item() 

    return wer_score



