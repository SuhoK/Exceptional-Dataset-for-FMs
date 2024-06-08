!pip install kobert-transformers
!pip install gluonnlp pandas tqdm
!pip install mxnet
!pip install sentence-transformers
  
import pandas as pd
import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizerFast, BertModel
from sentence_transformers import util

def calculate_sentence_similarity(pred_path,gt_path):
    df1 = pd.read_csv(pred_path)
    df2 = pd.read_csv(gt_path)

    model_name = 'kykim/bert-kor-base'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    def embed_sentence(sentence):
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().cpu()


    merged_df = pd.merge(df1, df2, on='filename', suffixes=('_file1', '_file2'))
    embeddings1 = [embed_sentence(sentence) for sentence in merged_df['content_file1']]
    embeddings2 = [embed_sentence(sentence) for sentence in merged_df['content_file2']]

    similarities = [util.pytorch_cos_sim(e1.unsqueeze(0), e2.unsqueeze(0)).item() for e1, e2 in zip(embeddings1, embeddings2)]
    merged_df['similarity'] = similarities

    return merged_df
