!pip install pandas torchmetrics
from torchmetrics.text import CharErrorRate
import pandas as pd

def recall_precision_f1("predicted.csv"):
    df1 = pd.read_csv("\predicted.csv")
    df2 = pd.read_csv("\gt.csv")

    # Recall 및 Precision 계산 함수
    def recall_and_precision(row):
        gpt_content = row['content_gpt']
        ocr_content = row['content_ocr']
        
        len_gpt = len(gpt_content)
        len_ocr = len(ocr_content)

        # 일치하는 글자 수 계산
        matches = sum(1 for gpt_char, ocr_char in zip(gpt_content, ocr_content) if gpt_char == ocr_char)

        # Recall 및 Precision 계산
        recall = matches / len_gpt if len_gpt > 0 else 0
        precision = matches / len_ocr if len_ocr > 0 else 0
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return pd.Series([recall, precision,f1])

def calculate_cer_from_csv(preds_csv_path, gt_csv_path):
    # CSV 파일 읽기
    preds_df = pd.read_csv(preds_csv_path)
    gt_df = pd.read_csv(target_csv_path)

    # filename을 기준으로 데이터 매칭
    merged_df = pd.merge(preds_df, target_df, on='filename', suffixes=('_pred', '_gt'))

    # 예측값과 타겟값 리스트 생성
    preds = merged_df[preds_column + '_pred'].tolist()
    gt = merged_df[target_column + '_gt'].tolist()

    # WER 계산
    cer = CharErrorRate()
    cer_score = cer(preds, gt).item()  # .item()을 사용하여 텐서 값을 가져옵니다.

    return cer_score
