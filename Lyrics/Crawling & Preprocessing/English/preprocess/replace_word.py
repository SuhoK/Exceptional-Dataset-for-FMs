import csv

def replace_word(input_file_path:str, output_file_path:str, replace:list=None) -> None:
    """
        replace several letters in csv file
        replace: 
        input_file_path: str, input file name
        output_file_path: str, output file name
    """
    replace = [['é', 'e'], ['ñ', 'n'], ['ó', 'o'], ['í', 'i'], ['Ó', 'O'], [' *', ''], ['&', 'and'],
                ['ú', 'u'], ['Á', 'A'], ['Ú', 'U'], ['ü', 'u'], ['ö', 'o'], [' X ', ' and '], [' @', ''],
                ['B52s', 'B 52s'], ['.', ''], ["¥$: Ye and Ty Dolla $ign", "¥$ Kanye West and Ty Dolla $ign"], ['$ign', 'sign'],
                ['(', ''], [')', ''], ['!', ''], ['?', ''], ['’', ''], [',', ''], ["'", ""], ['+', ''], ['@', '']]

    data = []
    with open(input_file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            data.append(row)

    for replace_pair in replace:
        old_word, new_word = replace_pair
        
        for i in range(len(data)):
            data[i]['Title'] = data[i]['Title'].replace(old_word, new_word)
            data[i]['Artist'] = data[i]['Artist'].replace(old_word, new_word)

    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Title', 'Artist']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)