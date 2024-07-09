import csv

def read_file(file_name):
    # save the description in a csv file with the format: Title, Artist, Description
    songs_data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            songs_data.append(row)
    return songs_data

def modify_genre(file_path: str, dest_path: str) -> None:
    songs_data = read_file(file_path)

    new_data = []
    for song in songs_data:
        genre_list = song['Genre'][1:-1].split(', ')
        for i in range(len(genre_list)):
            genre_list[i] = genre_list[i].lower()
            genre_list[i] = genre_list[i].replace('-', ' ')
        song['Genre'] = '[' + ', '.join(genre_list) + ']'

        new_data.append(song)

    with open(dest_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Title', 'Artist', 'lyrics', 'description', 'Genre'])
        writer.writeheader()
        writer.writerows(new_data)

if __name__ == '__main__':
    pass