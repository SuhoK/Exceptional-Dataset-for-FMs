import pandas as pd

def delete_duplicate(dir: str = 'English', file1_path: str = 'yearly_unilingual.csv', 
                     file2_path: str = 'weekly_unilingual.csv') -> None:
    # Load the data from the two files
    file1 = pd.read_csv(f'{dir}/{file1_path}')
    file2 = pd.read_csv(f'{dir}/{file2_path}')

    # Compare the two files and remove the duplicates from the second file if exist. 
    # Compare based on Title and Artist
    file2_cleaned = file2[~file2[['Title', 'Artist']].apply(tuple, 1).isin(file1[['Title', 'Artist']].apply(tuple, 1))]

    # Save the cleaned file
    file2_cleaned.to_csv(f'{dir}/{file2_path}', index=False)

    print(f'{len(file2_cleaned)} left in {dir}/{file2_path} after removing duplicates.')

if __name__ == '__main__':
    delete_duplicate()