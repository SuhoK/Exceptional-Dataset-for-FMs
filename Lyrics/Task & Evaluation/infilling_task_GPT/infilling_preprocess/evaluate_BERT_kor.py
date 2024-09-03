import torch
import os
from rouge import Rouge
from korouge_score import rouge_scorer
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from sklearn.metrics.pairwise import cosine_similarity

def get_embeddings(text_list, tokenizer, model):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.hidden_states[-1].mean(dim=1).detach().numpy()
    return embeddings

def evaluate_using_cosine_similarity(inputs_to_text, model_predicted_text, model, tokenizer):
    input_list = inputs_to_text.split('\n')
    predicted_list = model_predicted_text.split('\n')
    input_embedding = get_embeddings(input_list, tokenizer, model)
    predicted_embedding = get_embeddings(predicted_list, tokenizer, model)

    scores = cosine_similarity(predicted_embedding, input_embedding)
    return scores

def get_file_names(directory):
    # Get all the name of files in the directory
    # if directory is not directory return nothing
    if not os.path.isdir(directory):
        return []
    file_names = os.listdir(directory)
    file_names.sort()
    return file_names

def read_file(file_name):
    # Get the text from the file
    text = ''
    with open(file_name, 'r') as f:
        for line in f:
            text += line + ' '

    return text

def predicted_text(outputs, tokenizer):
    predicted_text = ''
    logits = outputs.logits
    softmax = torch.nn.Softmax(dim=-1)
    for i in range(logits.size(1)):
        mask_word_logits = softmax(logits[0, i])
        predicted_index = torch.argmax(mask_word_logits).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        predicted_text += (tokenizer.convert_tokens_to_string(predicted_token) + ' ')
    
    return predicted_text.replace(' ##', '')

def predict_with_BERT(text, masked_text, tokenizer, model):
    # Using BERT, evaluate the performance of the model on the test set
    
    origin = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs = tokenizer(masked_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    
    inputs['labels'] = origin['input_ids']

    with torch.no_grad():
        outputs = model(**inputs)

    return origin, outputs

def write_file(file_name, text):
    # Write the text to the file
    with open(file_name, 'w') as f:
        f.write(text)

def evaluate_using_rouge(text, predicted_text):
    # Evaluate the performance of the model using ROUGE
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    model_score = rouge.score(text, predicted_text)
    return model_score

def evaluate_BERT_kor():
    file_names = get_file_names('./final_dataset/korean_masking_task_2024')

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

    config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased', config=config)

    count = 0
    total_score = 0.0
    score_list = [0, 0, 0, 0, 0]

    for file_name in file_names:
        dir_name = f'./final_dataset/korean_masking_task_2024/{file_name}'
        if not get_file_names(dir_name) == []:
            masked_url = f'{dir_name}/masked.txt'
            masked_text = read_file(masked_url)
            
            clean_url = f'{dir_name}/original.txt'
            text = read_file(clean_url)

            inputs, outputs = predict_with_BERT(text, masked_text, tokenizer, model)
            model_predicted_text = predicted_text(outputs, tokenizer)
            print(model_predicted_text)

            inputs_to_text = ''.join(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][1:-1])))

            cosine_similarity_score_int = evaluate_using_cosine_similarity(inputs_to_text, model_predicted_text, model, tokenizer)[0][0]
            cosine_similarity_score = str(cosine_similarity_score_int)

            rouge_score = evaluate_using_rouge(inputs_to_text, model_predicted_text)
            
            total_score += cosine_similarity_score_int

            print(file_name)
            write_file(f'{dir_name}/bert_predict_scores.txt', f'{rouge_score}\n{cosine_similarity_score}')
if __name__ == '__main__':
    pass