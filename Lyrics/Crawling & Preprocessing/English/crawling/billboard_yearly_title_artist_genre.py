from tqdm import tqdm
from bs4 import BeautifulSoup
import requests as r
import csv

def get_billboard_songs_title_singer(year: int, dir_to_write: str = 'song_info') -> None:
    url = f'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}'
    response = r.get(url)
    wiki_page_text = response.text

    #create a BesutifulSoup Object
    soup = BeautifulSoup(wiki_page_text, 'html.parser')

    songs_table = soup.find('table',{'class':'wikitable sortable'})

    rows = songs_table.find_all('tr')[1:]
    songs_data = []

    prev_col = None
    for row in rows:
        columns = row.find_all('td')
        
        title = columns[1].text.strip().replace('"', '')
        if len(columns) == 3:
            artist = columns[2].text.strip()
        else:
            artist = prev_col[2].text.strip()

        # delete the letters that are not alphabet or number
        title = ''.join(e for e in title if e.isalnum() or e.isspace())
        artist = ''.join(e for e in artist if e.isalnum() or e.isspace())
        songs_data.append({"Title": title, "Artist": artist})

        prev_col = columns

    with open(f'{dir_to_write}/{year}.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Title', 'Artist']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(songs_data)

def extract_genre(table: BeautifulSoup) -> list:
    genres = []
    for row in table.find_all('tr'):
        header = row.find('th')
        if header and 'Genre' in header.text:
            genre_list = row.find('td').find_all('a')
            for genre in genre_list:
                # if genre does not starts with '[', save it to the genres list
                if not genre.text.startswith('['):
                    genres.append(genre.text)
    return genres

def get_soup(url: str) -> tuple:
    response = r.get(url)
    wiki_page_text = response.text
    soup = BeautifulSoup(wiki_page_text, 'html.parser')

    return response, soup

def append_songs_data_genre(genres: list, songs_data_genre: list, song: dict, no_genre: int) -> tuple:
    if len(genres) != 0:
        song['Genre'] = genres
        songs_data_genre.append(song)
    else:
        no_genre += 1

    return songs_data_genre, no_genre

def need_artist(song: dict, title: str, songs_data_genre: list, no_genre: int, failed_response: int) -> tuple:
    # no table, then change the url
    artist = song['Artist'].split()
    artist = '_'.join(artist)
    url = f'https://en.wikipedia.org/wiki/{title}_({artist}_song)'
    response, soup = get_soup(url)
    if response.status_code == 200:
        table = soup.find('table', class_=lambda x: x and x.startswith('infobox vevent'))
        if table and table.find_all('tr'):
            genres = extract_genre(table)
            songs_data_genre, no_genre = append_songs_data_genre(genres, songs_data_genre, song, no_genre)
        else:
            # give up
            pass
    else:
        failed_response += 1
    
    return songs_data_genre, no_genre, failed_response

def get_billboard_songs_genre(title_singer_file: str='billboard_songs_all_with_lyrics_description.csv'
                              , file_to_write: str='billboard_songs_all_with_lyrics_description_genre.csv'
                              , fieldnames: list=['Title', 'Artist', 'lyrics', 'description', 'Genre']) -> tuple:
    # read billboard_songs_all_with_lyrics_description.csv file
    songs_data = []
    with open(title_singer_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            songs_data.append(row)

    failed_response = 0         # count the number of wrong url
    no_genre = 0                # count the number of songs that do not have genre
    songs_data_genre = []       # List to save it to csv file

    # Get the genre of the song
    for song in tqdm(songs_data):
        # make url
        title = song['Title'].split()
        
        replace_words = [[" dont "," don't "], [" cant ", " can't "], [" wont ", " won't "], [" aint ", "ain't"], [" Dont "," Don't "], [" Cant ", " Can't "], [" Wont ", " Won't "], [" Aint ", "Ain't"], [" Im ", " I'm "], [" youre ", " you're "],[" Ill ", " I'll "], [" thats ", " that's "],[" whats ", " what's "],[" Thats ", " That's "],[" Whats ", " What's "]]
        for replace_word in replace_words:
            title = [t.replace(replace_word[0], replace_word[1]) for t in title]
        
        title = '_'.join(title)
        url = f'https://en.wikipedia.org/wiki/{title}'

        # Crawling
        response, soup = get_soup(url)
        if response.status_code == 200:
            # find the table
            table = soup.find('table', class_=lambda x: x and x.startswith('infobox vevent'))

            # find the genre in table
            if table and table.find_all('tr'):
                genres = extract_genre(table)
                # if genre exists, save it to the sonts_data_genre list
                # input: genres, songs_data_genre, song, no_genre
                songs_data_genre, no_genre = append_songs_data_genre(genres, songs_data_genre, song, no_genre)

            else:
                # no table, then change the url
                url = f'https://en.wikipedia.org/wiki/{title}_(song)'
                response, soup = get_soup(url)
                if response.status_code == 200:
                    table = soup.find('table', class_=lambda x: x and x.startswith('infobox vevent'))
                    if table and table.find_all('tr'):
                        genres = extract_genre(table)
                        songs_data_genre, no_genre = append_songs_data_genre(genres, songs_data_genre, song, no_genre)
                    else:
                        need_artist(song, title, songs_data_genre, no_genre, failed_response)
                else:
                    need_artist(song, title, songs_data_genre, no_genre, failed_response)
        else:
            failed_response += 1
    
    print(f"{len(songs_data_genre)} data left after crawling genre.")
    with open(file_to_write, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(songs_data_genre)
    
    return failed_response, no_genre

if __name__ == '__main__':
    for year in range(1990, 2024):
        get_billboard_songs_title_singer(year)
    print(get_billboard_songs_genre())