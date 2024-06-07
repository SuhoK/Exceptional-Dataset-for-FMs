# from write_paper_BERT_evaluate import evaluate_infilling_tasks_ENG
from divide_lyrics import divide_lyrics
from masking_lyrics_english import mask_all
from evaluate_BERT_eng import evaluate_bert_eng
from get_list_for_gpt import get_list_for_gpt
from get_response_gpt import get_response_eng
from evaluate_gpt import evaluate_gpt_eng

if __name__ == '__main__':
    index = divide_lyrics('English/yearly_unilingual_final.csv')
    divide_lyrics('English/weekly_unilingual_final.csv', index+1)

    mask_all()
    evaluate_bert_eng()

    get_list_for_gpt('./final_dataset/English_masking_task_words', 0.9)
    get_response_eng("your own API Key", './final_Dataset/English_masking_task_words')

    evaluate_gpt_eng()
