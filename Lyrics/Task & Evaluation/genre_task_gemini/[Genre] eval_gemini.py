import pandas as pd

# Function to clean and split genre strings
def clean_and_split_genres(genre_str):
    genre_str = genre_str.replace('[', '').replace(']', '').replace("'", "").replace("Genres: ", "").replace("Genre: ", "").replace("\n", "").replace("**", "")
    genre_list = genre_str.split(', ')
    print(genre_list)
    real_genre_list = []
    for genre in genre_list:
        real_genre_list.append(genre.replace(" ", ""))
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

merged_df = pd.read_csv('Gemini_genre_result_ENG_yearly_#1_200.csv')

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
for _, row in merged_df.iterrows():
    true_genre = row['Genre']  # Assuming the ground truth genre column is named 'Genre'
    
    # Zero-shot
    predicted_genre = row['Zero-shot']
    total_metrics['zero_shot_overlap_ratio'] += calculate_overlap_ratio(predicted_genre, true_genre)
    total_metrics['zero_shot_exact_match'] += calculate_exact_match(predicted_genre, true_genre)
    print(total_metrics['zero_shot_overlap_ratio'], total_metrics['zero_shot_exact_match'])
    
    # Chain-of-thought
    predicted_genre = row['COT']
    total_metrics['cot_overlap_ratio'] += calculate_overlap_ratio(predicted_genre, true_genre)
    total_metrics['cot_exact_match'] += calculate_exact_match(predicted_genre, true_genre)
    
    # Chain-of-thought few-shot
    predicted_genre = row['COT_few_shot']
    total_metrics['cot_few_shot_overlap_ratio'] += calculate_overlap_ratio(predicted_genre, true_genre)
    total_metrics['cot_few_shot_exact_match'] += calculate_exact_match(predicted_genre, true_genre)
    
    total_metrics['total_rows'] += 1

# Calculate final averages
total_average_metrics = {key: value / total_metrics['total_rows'] for key, value in total_metrics.items() if key != 'total_rows'}

# Append final summary row
results_summary.append({
    'Zero-shot Overlap Ratio': total_average_metrics['zero_shot_overlap_ratio'],
    'Zero-shot Exact Match': total_average_metrics['zero_shot_exact_match'],
    'CoT Overlap Ratio': total_average_metrics['cot_overlap_ratio'],
    'CoT Exact Match': total_average_metrics['cot_exact_match'],
    'CoT Few-shot Overlap Ratio': total_average_metrics['cot_few_shot_overlap_ratio'],
    'CoT Few-shot Exact Match': total_average_metrics['cot_few_shot_exact_match']
})

# Create summary DataFrame
results_summary_df = pd.DataFrame(results_summary)

# Save the results to a new CSV file
results_summary_df.to_csv('Gemini_genre_result_eval.csv', index=False)

print(results_summary_df)


