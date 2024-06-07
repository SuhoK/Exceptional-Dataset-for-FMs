import pandas as pd
import os
from tqdm import tqdm


def make_directory(title: str) -> None:
    """
    Create a directory for the text files.
    """
    # Define the directory name
    directory = f'./lyrics_only_txt{title}'

    # Create the directory
    os.makedirs(directory, exist_ok=True)

def divide_lyrics(data_path: str, song_index: int = 0) -> int:
    """
        Extracting lyrics from the CSV file and creating text files for each song parts.
    """

    df = pd.read_csv(data_path)
    make_directory('')

    df.columns = df.columns.str.strip()

    for index, row in tqdm(df.iterrows()):
        # Extract the title and lyrics
        title = row['Title'] # Remove any surrounding whitespace
        lyrics = row['lyrics']

        make_directory(f'/{song_index}_{title}')
        # Divide lyrics when '[' appears

        if len(lyrics.split('[')) == 1:
            filename = f'./lyrics_only_txt/{song_index}_{title}/0.txt'
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(lyrics)
        else:
            lyric_set = lyrics.split('[')[1:]

            index_lyrics = 0
            for lyric in lyric_set:
                # Divide lyrics by ']' and take the second part
                lyrics_lyric = lyric.split(']')[1]
                # Define the filename
                filename = f'./lyrics_only_txt/{song_index}_{title}/{index_lyrics}.txt'

                # Write the lyrics to a text file
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(lyrics_lyric)
                index_lyrics += 1
        song_index += 1

    # Provide feedback that the process is complete
    return song_index

if __name__ == '__main__':
    divide_lyrics('English/yearly_unilingual_final.csv')