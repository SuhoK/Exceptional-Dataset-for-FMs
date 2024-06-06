!pip install pandas torchmetrics
from torchmetrics.text import CharErrorRate
import pandas as pd

def recall_precision_f1("predicted.csv"):
    df1 = pd.read_csv("\predicted.csv")
    df2 = pd.read_csv("\gt.csv")

    def recall_and_precision(row):
        gpt_content = row['content_gpt']
        ocr_content = row['content_ocr']
        
        len_gpt = len(gpt_content)
        len_ocr = len(ocr_content)

        matches = sum(1 for gpt_char, ocr_char in zip(gpt_content, ocr_content) if gpt_char == ocr_char)
        recall = matches / len_gpt if len_gpt > 0 else 0
        precision = matches / len_ocr if len_ocr > 0 else 0
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return pd.Series([recall, precision,f1])

def cer(preds_csv_path, gt_csv_path):
    preds_df = pd.read_csv(preds_csv_path)
    gt_df = pd.read_csv(target_csv_path)
    merged_df = pd.merge(preds_df, target_df, on='filename', suffixes=('_pred', '_gt'))
    preds = merged_df[preds_column + '_pred'].tolist()
    gt = merged_df[target_column + '_gt'].tolist()


    cer = CharErrorRate()
    cer_score = cer(preds, gt).item()  

    return cer_score

def LDMETRIC(preds_csv_path, gt_csv_path):
    df1 = pd.read_csv(preds_csv_path)
    df2 = pd.read_csv(gt_csv_path)
    merged_df = pd.merge(df1, df2, on='filename', suffixes=('_gpt', '_ocr'))


    def levenshtein_distance(row):
        return distance(row['content_gpt'], row['content_ocr'])

    merged_df['levenshtein_distance'] = merged_df.apply(levenshtein_distance, axis=1)
    average_ld = merged_df['levenshtein_distance'].mean()

    return average_ld

