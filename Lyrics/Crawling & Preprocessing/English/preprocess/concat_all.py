import csv

def concat_all(dir:str='English', file_names:list = range(1990, 2024), 
               fieldnames:list=['Title', 'Artist'], write_file:str='total.csv'
               ):
    data = []
    for file_name in file_names:
        with open(f'{dir}/{file_name}.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                flag = True

                # check if the row already exists in the data
                for data_row in data:
                    if row['Title'] == data_row['Title'] and row['Artist'] == data_row['Artist']:
                        flag = False
                        break
                
                # Append the row only if it is not duplicate
                if flag:
                    data.append(row)
    print(f"{len(data)} data left after concatenation.")
    
    with open(f'{dir}/{write_file}', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
