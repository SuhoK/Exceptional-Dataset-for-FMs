from infilling_preprocess.divide_lyrics import divide_lyrics
from infilling_preprocess.masking_lyrics_english import mask_all
from infilling_preprocess.evaluate_BERT_eng import evaluate_bert_eng
from infilling_preprocess.get_list_for_gpt import get_list_for_gpt

from infilling_get_response_gemini import infilling_Eng
from evaluate_eng import evaluate_eng

if __name__ == '__main__':
    index = divide_lyrics('English/yearly_unilingual_final.csv')
    divide_lyrics('English/weekly_unilingual_final.csv', index+1)

    mask_all()
    evaluate_bert_eng()

    get_list_for_gpt('./final_dataset/English_masking_task_words', 0.9)
    infilling_Eng("your own API Key", './final_Dataset/English_masking_task_words')

    evaluate_eng()
