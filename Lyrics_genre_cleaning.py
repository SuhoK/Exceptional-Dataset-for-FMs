import pandas as pd
import csv

def read_file(file_name):
    # save the description in a csv file with the format: Title, Artist, Description
    songs_data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            songs_data.append(row)
    return songs_data

def make_filtered_genre_list(dir: str, yearly_data: list, weekly_data: list) -> str:
    # Get the list of genres
    genre_dict_list = {}
    final_genre_list = []
    for song in yearly_data + weekly_data:
        genre_list = song['Genre'][1:-1].replace("'", "").split(', ')
        for genre in genre_list:
            genre = genre.strip()
            if genre in genre_dict_list:
                genre_dict_list[genre] += 1
                if genre_dict_list[genre] >= 10 and genre not in final_genre_list:
                    final_genre_list.append(genre)
            else:
                genre_dict_list[genre] = 1
    
    final_genre_dict = []
    for genre in final_genre_list:
        genre_dict = {'Genre':genre, 'Count': genre_dict_list[genre]}
        final_genre_dict.append(genre_dict)

    print(f'{len(final_genre_dict)} genres found')
    # Save the filtered genre list to a CSV file
    filtered_genre_list_path = f'{dir}/filtered_genre_list.csv'
    with open(filtered_genre_list_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Genre', 'Count'])
        writer.writeheader()
        writer.writerows(final_genre_dict)
    
    return filtered_genre_list_path

def clean_and_filter_genres(genre_str, filtered_genres):
    # Remove brackets and split by comma
    genres = genre_str.strip("[]").replace("'", "").split(", ")
    # Filter genres
    filtered = [genre for genre in genres if genre in filtered_genres]
    return filtered

def genre_cleaning(dir: str = 'English', file_path_yearly: str = 'yearly_unilingual.csv', 
                   file_path_weekly: str = 'weekly_unilingual.csv') -> None:
    # Load the yearly and weekly data
    yearly_data = read_file(f'{dir}/{file_path_yearly}')
    weekly_data = read_file(f'{dir}/{file_path_weekly}')
    filtered_genre_path = make_filtered_genre_list(dir, yearly_data, weekly_data)

    # Load the filtered genre counts CSV
    filtered_genre_counts = pd.read_csv(filtered_genre_path)

    # Get the list of filtered genres
    filtered_genres = filtered_genre_counts['Genre'].tolist()

    file1_data = pd.read_csv(f'{dir}/{file_path_yearly}')
    file2_data = pd.read_csv(f'{dir}/{file_path_weekly}')

    file1_data['Genre'] = file1_data['Genre'].apply(lambda x: clean_and_filter_genres(x, filtered_genres))
    file2_data['Genre'] = file2_data['Genre'].apply(lambda x: clean_and_filter_genres(x, filtered_genres))

    file1_data = file1_data[file1_data['Genre'].map(len) > 0]
    file2_data = file2_data[file2_data['Genre'].map(len) > 0]

    final_rows_file1 = len(file1_data)
    final_rows_file2 = len(file2_data)

    filtered_file1_path = f'{dir}/{file_path_yearly[:-4]}_final.csv'
    filtered_file2_path = f'{dir}/{file_path_weekly[:-4]}_final.csv'

    file1_data.to_csv(filtered_file1_path, index=False)
    file2_data.to_csv(filtered_file2_path, index=False)

    print(f'Number of remaining rows in {file_path_yearly}.csv: {final_rows_file1}')
    print(f'Number of remaining rows in {file_path_weekly}.csv: {final_rows_file2}')

if __name__ == '__main__':
    genre_cleaning()