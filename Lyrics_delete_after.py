import csv

def delete_after(row_num: int, word: str, input_file_path: str='song_billboard/billboard_weekly_songs_all.csv', 
                 output_file_path: str='song_billboard/billboard_weekly_songs_all.csv') -> None:
    """
        Delete all text after the word
        word: str, the word to delete after
        input_file_path: str, input file name
        output_file_path: str, output file name
    """
    # Open the input CSV file
    with open(input_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Read all data from the CSV file
        rows = list(reader)

    # List to hold cleaned data
    cleaned_rows = []

    # Process each row
    for row in rows:
        if word in row[row_num]:  # Assuming 'featuring' is in the second column
            # Split on 'featuring' and keep the part before it
            row[row_num] = row[row_num].split(word)[0].strip()
        cleaned_rows.append(row)

    # Write cleaned data to a new CSV file
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(cleaned_rows)

    print("Cleaned data has been written to", output_file_path)

if __name__ == '__main__':
    weekly = ['20240125_20240104', '20240229_20240208', '20240328_20240307', '20240425_20240404']
    for year in range(1990, 2024):
        delete_after('featuring', year)