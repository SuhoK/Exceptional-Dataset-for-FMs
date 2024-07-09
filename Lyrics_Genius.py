from bs4 import BeautifulSoup
import requests as r
import csv
from tqdm import tqdm

def read_file(file_name: str) -> list:
    # save the description in a csv file with the format: Title, Artist, Description
    songs_data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            songs_data.append(row)

    return songs_data

def make_url(songs_data: list) -> list:
    url_list = []
    for song in songs_data:
        artist = song['Artist']
        title = song['Title']
        artist = artist.split()
        title = title.split()
        if len(artist) > 1:
            artist = artist[0] + '-' + '-'.join(artist[1:]).lower()
        else:
            artist = artist[0]
        title = '-'.join(title).lower()
        url = f"https://genius.com/{artist}-{title}-lyrics"
        url_list.append(url)
    return url_list

def get_lyrics(url: str) -> str:
    response = r.get(url)
    genius_page_text = response.text
    soup = BeautifulSoup(genius_page_text, 'html.parser')
    if response.status_code == 200:
        # There can exist multiple <p> tags in this div. I have to get all <p> tags and concatenate them.
        main_div = soup.find('div', {"class": "PageGriddesktop-a6v82w-0 SongPageGriddesktop-sc-1px5b71-0 Lyrics__Root-sc-1ynbvzw-0 iEyyHq"}).find_all('div', attrs={"data-lyrics-container": "true"})
        all_lyrics = []
        for div in main_div:
            for be in div.find_all('br'):
                be.replace_with('\n')
            lyrics_text = div.text
            all_lyrics.append(lyrics_text)
        return '\n'.join(all_lyrics)
    else:
        return 1

def get_description(url: str) -> str:
    response = r.get(url)
    genius_page_text = response.text
    soup = BeautifulSoup(genius_page_text, 'html.parser')
    if response.status_code == 200:
        main_div = soup.find('div', {"class": "ExpandableContent__Container-sc-1165iv-0 ikywhQ"})
        description = main_div.text
        if description == '':
            return 1
        return description
    else:
        return 1

def crawl_lyrics_description(songs_url: str, songs_result_url: str) -> None:
    # Read the file and make the url
    songs_data = read_file(songs_url)
    url_list = make_url(songs_data)

    # Get the lyrics for each song
    # use tqdm to see the progress
    song_result = []
    for i in (range(len(songs_data))):
        lyrics = get_lyrics(url_list[i])
        if lyrics != 1:
            songs_data[i]['lyrics'] = lyrics

        description = get_description(url_list[i])
        if description != 1:
            songs_data[i]['description'] = description
        
        if lyrics != 1 and description != 1 and not description.lower().startswith("credits"):
            song_result.append(songs_data[i])
        else:
            print(f"Couldn't get lyrics or description for {songs_data[i]['Title']} by {songs_data[i]['Artist']}")

    print(f"{len(song_result)} songs left after crawling lyrics and description.")
    with open(songs_result_url, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Title', 'Artist', 'lyrics', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(song_result)