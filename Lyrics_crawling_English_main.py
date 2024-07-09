from crawling.billboard_yearly_title_artist_genre import get_billboard_songs_title_singer, get_billboard_songs_genre
from crawling.Genius import crawl_lyrics_description
from crawling.billboard_weekly import crawl_billboard_weekly, concatenate_files
from preprocess.delete_after import delete_after
from preprocess.replace_word import replace_word
from preprocess.concat_all import concat_all

from preprocess.delete_multilingual import delete_multilingual
from preprocess.modify_genre import modify_genre
from preprocess.delete_duplicate import delete_duplicate
from preprocess.genre_cleaning import genre_cleaning


def yearly_crawling(dir: str = 'English') -> None:
    # Takes about 4 hours
    print("\nCrawling weekly songs...")
    # Crawling songs and title
    for year in range(1990, 2024):
        get_billboard_songs_title_singer(year, dir)
        print(f"Crawling songs and title for year {year} is done. Saved in {dir}/{year}.csv")

    # integrate yearly data
    concat_all(dir)
    print(f"Integration is done. Saved in {dir}/total.csv")

    # Preprocessing
    delete_after(0, 'featuring', f'{dir}/total.csv', f'{dir}/total.csv')
    delete_after(1, 'featuring', f'{dir}/total.csv', f'{dir}/total.csv')
    replace_word(f'{dir}/total.csv', f'{dir}/total.csv')
    print(f"Preprocessing for all songs is done. Saved in {dir}/total.csv")

    # Crawling lyrics
    crawl_lyrics_description(f'{dir}/total.csv', f'{dir}/total.csv')
    print(f"Crawling lyrics and descriptions for all songs is done. Saved in {dir}/total.csv")

    # Crawling genre
    get_billboard_songs_genre(f'{dir}/total.csv', f'{dir}/total.csv')
    print(f"Crawling genre for all songs is done. Saved in {dir}/total.csv")

def weekly_crawling(dir: str = 'English') -> None:
    # Weekly crawling
    print("\nCrawling weekly songs...")
    file_names = crawl_billboard_weekly(dir)
    print(f"Crawling weekly songs is done. Saved in {dir}")

    # Concatenate weekly files
    concatenate_files(file_names, f'{dir}/total_weekly.csv')

    # Preprocessing
    delete_after(1, 'with', f'{dir}/total_weekly.csv', f'{dir}/total_weekly.csv')
    delete_after(1, 'Featuring', f'{dir}/total_weekly.csv', f'{dir}/total_weekly.csv')

    replace_word(f'{dir}/total_weekly.csv', f'{dir}/total_weekly.csv')
    print(f"Preprocessing for all songs is done. Saved in {dir}/total_weekly.csv")
    
    # Crawling lyrics
    crawl_lyrics_description(f'{dir}/total_weekly.csv', f'{dir}/total_weekly.csv')
    print(f"Crawling lyrics and descriptions for all songs is done. Saved in {dir}/total_weekly.csv")

    # Crawling genre
    get_billboard_songs_genre(f'{dir}/total_weekly.csv', f'{dir}/total_weekly.csv')
    print(f"Crawling genre for all songs is done. Saved in {dir}/total_weekly.csv")

if __name__ == '__main__':
    dir = 'English'
    yearly_crawling(dir)
    weekly_crawling(dir)

    delete_multilingual('English/total.csv', 'English/yearly_unilingual.csv')
    delete_multilingual('English/total_weekly.csv', 'English/weekly_unilingual.csv')

    delete_duplicate('English', 'yearly_unilingual.csv', 'weekly_unilingual.csv')

    modify_genre('English/yearly_unilingual.csv', 'English/yearly_unilingual.csv')
    modify_genre('English/weekly_unilingual.csv', 'English/weekly_unilingual.csv')

    genre_cleaning('English', 'yearly_unilingual.csv', 'weekly_unilingual.csv')
