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

# csv file with predicted sentences and ground truth
pred_csv = pd.read_csv("/gpt_cot.csv")
gt_csv = pd.read_csv("/label_ocr.csv")
merged_df = pd.merge(pred_csv, gt_csv, on='filename', suffixes=('_pred', '_gt'))
pred_li = merged_df['content_pred'].apply(clean_text).tolist()
gt_li = merged_df['content_gt'].apply(clean_text).tolist()
accuracies = [accuracy(p.split(), t.split()) for p, t in zip(pred_li, gt_li)]

# Calculate
prec = precision(pred_li, gt_li)
rec = recall(pred_li, gt_li)
f1 = f1_score(prec, rec)
acc = sum(accuracies) / len(accuracies)
