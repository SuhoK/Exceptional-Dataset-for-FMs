from bs4 import BeautifulSoup
import pandas as pd
import requests as r
import calendar
import csv
import os

def crawl_billboard(url:str) -> list:
    response = r.get(url)
    billboard_text = response.text
    soup = BeautifulSoup(billboard_text, 'html.parser')

    # Find the main container
    charts = soup.find_all('div', class_='o-chart-results-list-row-container')

    # Initialize lists to store the data
    music_info = []

    # Loop through every entry in the chart
    for chart in charts:
        title_tag = chart.find('h3', class_=lambda x: x and x.startswith('c-title a-no-trucate'))
        artist_tag = chart.find('span', class_=lambda x: x and x.startswith('c-label a-no-trucate'))

        if title_tag and artist_tag:
            title = title_tag.get_text(strip=True)
            artist = artist_tag.get_text(strip=True)
            music_info.append({'Title': title, 'Artist': artist})
    return music_info

def write_file(dir:str, df: dict, date: int) -> None:
    with open(f'{dir}/weekly_{date}.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Title', 'Artist']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(df)

def crawl_and_save(start_date: int, dir: str) -> list:
    month = start_date // 100 % 100  # Extract the month from the start date
    _, num_days = calendar.monthrange(start_date // 10000, month)  # Get the number of days in the month

    # monthly date 개수가 30 or 31이라는 점 반영
    date = []
    current_date = start_date
    while current_date % 100 < 32 and current_date % 100 != 0:
        # 20240425 => 2024-04-25
        current_date_str = str(current_date)
        current_date_str = f'{current_date_str[:4]}-{current_date_str[4:6]}-{current_date_str[6:]}'
        url = f'https://www.billboard.com/charts/hot-100/{current_date_str}'
        df = crawl_billboard(url)
        write_file(dir, df, current_date)
        date.append(current_date)
        current_date -= 7  # Move to the previous week
    return date

def crawl_billboard_weekly(dir: str) -> list:
    start_dates = [20240425, 20240328, 20240229, 20240125]

    file_names = []
    for i in range(len(start_dates)):
        dates = crawl_and_save(start_dates[i], dir)
        for date in dates:
            file_names.append(f'{dir}/weekly_{date}.csv')
    
    return file_names

def concatenate_files(file_names: list, output_path: str) -> None:
    all_rows = []
    for file_name in file_names:
        df = pd.read_csv(file_name)
        all_rows.append(df)
    
    # Delete duplicates and save it into csv
    combined_df = pd.concat(all_rows, ignore_index=True)
    cleaned_df = combined_df.drop_duplicates(keep='first')
    print(f"Number of songs after concatenate: {len(cleaned_df)}")
    cleaned_df.to_csv(output_path, index=False, encoding='utf-8')

if __name__ == '__main__':
    # # Crawling from billboard
    # crawl_main()

    # # Delete duplicated rows and concat all csv files
    # file_names = get_all_filenames_in_directory('./song_billboard')
    # concatenate_files(file_names)
    
    # delete_after('Featuring')
    
    # print(get_billboard_songs_genre('./song_billboard/billboard_weekly_songs_all.csv',
    #                                 './song_billboard/billboard_weekly_songs_all_genre.csv',
    #                                 ['Title', 'Artist', 'Genre']))

    

    
    pass
