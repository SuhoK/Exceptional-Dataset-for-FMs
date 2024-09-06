import pandas as pd

# Function to clean and split genre strings
def clean_and_split_genres(genre_str):
    if not isinstance(genre_str, str):
        genre_str = ""  # string이 아닌 경우 비우기
    
    genre_str = genre_str.replace('[', '').replace(']', '').replace("'", "").replace("Genres: ", "").replace("Genre: ", "").replace("\n", "").replace("**", "")
    genre_list = genre_str.split(', ')
    real_genre_list = [genre.replace(" ", "") for genre in genre_list]
    return set(real_genre_list)

# Function to calculate overlap ratio
def calculate_overlap_ratio(predicted_genres, true_genres):
    predicted_set = clean_and_split_genres(predicted_genres)
    true_set = clean_and_split_genres(true_genres)
    intersection = predicted_set.intersection(true_set)
    return len(intersection) / len(true_set) if true_set else 0

# Function to calculate exact match accuracy
def calculate_exact_match(predicted_genres, true_genres):
    predicted_set = clean_and_split_genres(predicted_genres)
    true_set = clean_and_split_genres(true_genres)
    return 1 if not predicted_set.isdisjoint(true_set) else 0


# Initialize results list
results_summary = []



result_1 = pd.read_csv(os.path.join(ROOT_DIR, "Claude_genre_result_yearly_ENG.csv"), header=0)
result_2 = pd.read_csv(os.path.join(ROOT_DIR, "Claude_genre_result_yearly_ENG_#2.csv"), header=0)
merged_df = pd.merge(result_1, result_2, how="outer")

# Initialize totals for averaging
total_metrics = {
    'zero_shot_overlap_ratio': 0,
    'zero_shot_exact_match': 0,
    'cot_overlap_ratio': 0,
    'cot_exact_match': 0,
    'cot_few_shot_overlap_ratio': 0,
    'cot_few_shot_exact_match': 0,
    'total_rows': 0
}


# Iterate through the merged dataframe and calculate accuracy
for _, row in merged_df_2.iterrows():
    true_genre = row['Genre']  # Assuming the ground truth genre column is named 'Genre'
    
    # Zero-shot
    predicted_genre = row['Zero-shot']
    total_metrics['zero_shot_overlap_ratio'] += calculate_overlap_ratio(predicted_genre, true_genre)
    total_metrics['zero_shot_exact_match'] += calculate_exact_match(predicted_genre, true_genre)
    
    # Chain-of-thought
    predicted_genre = row['COT']
    total_metrics['cot_overlap_ratio'] += calculate_overlap_ratio(predicted_genre, true_genre)
    total_metrics['cot_exact_match'] += calculate_exact_match(predicted_genre, true_genre)
    
    # Chain-of-thought few-shot
    predicted_genre = row['COT_few_shot']
    total_metrics['cot_few_shot_overlap_ratio'] += calculate_overlap_ratio(predicted_genre, true_genre)
    total_metrics['cot_few_shot_exact_match'] += calculate_exact_match(predicted_genre, true_genre)
    
    total_metrics['total_rows'] += 1

# Calculate final averages for the entire dataset
total_average_metrics = {
    'Zero-shot Overlap Ratio': total_metrics['zero_shot_overlap_ratio'] / total_metrics['total_rows'],
    'Zero-shot Exact Match': total_metrics['zero_shot_exact_match'] / total_metrics['total_rows'],
    'CoT Overlap Ratio': total_metrics['cot_overlap_ratio'] / total_metrics['total_rows'],
    'CoT Exact Match': total_metrics['cot_exact_match'] / total_metrics['total_rows'],
    'CoT Few-shot Overlap Ratio': total_metrics['cot_few_shot_overlap_ratio'] / total_metrics['total_rows'],
    'CoT Few-shot Exact Match': total_metrics['cot_few_shot_exact_match'] / total_metrics['total_rows']
}

# Append the final summary row with averaged metrics
results_summary.append(total_average_metrics)

# Create summary DataFrame
results_summary_df = pd.DataFrame(results_summary)

print(results_summary_df)