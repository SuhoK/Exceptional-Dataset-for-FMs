import csv
from tqdm import tqdm
from langdetect import detect

def read_file(file_name:str) -> list():
    # Read the billboard_songs_all_with_lyrics_description_genre.csv file
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        songs_data = []
        for row in reader:
            songs_data.append(row)

    return songs_data

def detect_language(text:str) -> str:
    try:
        return detect(text)
    except:
        return "Language detection failed"

def delete_multilingual(prev_file:str = 'English/total.csv',
                        new_file: str = 'English/non-multilingual') -> None:
    # Read the file
    songs_data = read_file(prev_file)
    songs_after_delete = []
    for song in tqdm(songs_data):
        lyrics = song['lyrics']
        return_val = detect_language(lyrics)
        if return_val == 'en':
            songs_after_delete.append(song)

    # Save the file
    with open(new_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Title', 'Artist', 'Genre', 'lyrics', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(songs_after_delete)
    print("Delete multilingual song done.")
    print("after delete, the number of songs: ", len(songs_after_delete))

if __name__ == '__main__':
    pass