from masking_lyrics_korean import mask_all
from evaluate_BERT_kor import evaluate_BERT_kor
from get_list_for_gpt import get_list_for_gpt
from get_response_gpt import get_response_kor
from evaluate_gpt import evaluate_gpt_kor

if __name__ == '__main__':
    mask_all()
    evaluate_BERT_kor()

    get_list_for_gpt('./final_dataset/English_masking_task_words', 0.9)
    get_response_kor("your own API Key", './final_Dataset/Korean_masking_task_words')

    evaluate_gpt_kor()
