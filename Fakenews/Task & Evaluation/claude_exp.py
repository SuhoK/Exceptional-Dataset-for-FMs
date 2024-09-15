from os import name
import pandas as pd
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score

# Load the data from CSV and JSON files
data_csv = pd.read_csv('../data.csv')

with open('path_to_zeroshot_res', 'r') as f:
    zero_res = json.load(f)
    
with open('path_to_cot_res', 'r') as f:
    cot_res = json.load(f)
    
with open('path_to_few_res', 'r') as f:
    few_res = json.load(f)


    
def extract_text(text, start_tag, end_tag):
    '''
    A function to extract text from [Title] to [Contents] without stopword removal
    
    Args:
    
    Returns:
        
    '''
    start_idx = text.find(start_tag) + len(start_tag)
    if (end_tag == "Last"):
        end_idx = len(text)
    else:
        end_idx = text.find(end_tag)
    extracted_text = text[start_idx:end_idx].strip()
    return extracted_text

def calculate_acc(mode : str, res : json):
    
    """
        A function to calculate the accuracy, precision, recall and AUC score of the zeroshot prompt
    
        Args:
            mode: str
                The mode to calculate the accuracy, precision, recall and AUC score. 
                The mode must be one of the following:
                    - 'zero'
                    - 'cot'
                    - 'few'
        
    """
    
    json_data = []
    
    for item in res:
        request = item.get(f'{mode}_prompt')
        response = item.get(f'{mode}_response')
        label = item.get('actual_label')
        
        if request and response:
            clean_text = extract_text(request, '[Title]', '[Contents]')
            contents = extract_text(request, '[Title]', 'Last')
            response_number = re.search(r'\d+', response)
            if response_number:
                json_data.append((clean_text, contents, int(response_number.group()))) # (title, content, response)

    json_df = pd.DataFrame(json_data, columns=['title', 'content', 'response']).drop_duplicates(subset=['title'])

    # Extract title text and labels from CSV, ensuring unique titles
    csv_data = []
    for index, (row, label) in enumerate(zip(data_csv.iloc[:, 0], data_csv['label'])):  # Assuming the text is in the first column and labels in the 'label' column
        clean_text = extract_text(row, '[Title]', '[Contents]')
        csv_data.append((index, clean_text, label))  # Include the index for filtering

    csv_df = pd.DataFrame(csv_data, columns=['index', 'title', 'label']).drop_duplicates(subset=['title'])

    # Define filter indices
    onion_filtered = [2, 5, 14, 24, 28, 30, 73, 99, 102, 119, 123, 135, 153, 165, 168, 176, 182, 206, 212, 228, 248, 262, 297, 322, 332, 336, 343, 362, 385, 405, 414, 430, 450, 537, 573, 583, 609, 610, 615, 629, 639, 703, 714, 717, 725, 745, 782, 821, 835, 923, 927, 988, 1000, 1004, 1011, 1016, 1027, 1053, 1055, 1059, 1115, 1145, 1155, 1159, 1165, 1222, 1234, 1239, 1251, 1258, 1266, 1271, 1317, 1329, 1402, 1418, 1429, 1445, 1456, 1470, 1471, 1492, 1525, 1529, 1588, 1599, 1606, 1610, 1629, 1640, 1648, 1661, 1663, 1672, 1692, 1715, 1727, 1731, 1734, 1742, 1743, 1747, 1756, 1783, 1811, 1813, 1873, 1895, 1924, 1952, 1976, 1990, 1991, 2000, 2024, 2028, 2044, 2086, 2089, 2117, 2119, 2129, 2156, 2167, 2171, 2200, 2203, 2208, 2254, 2285, 2302, 2320, 2333, 2382, 2399, 2419, 2469, 2470, 2472, 2492, 2501, 2565, 2586, 2605, 2622, 2637, 2643, 2656, 2658, 2742, 2746, 2764, 2808, 2823, 2834, 2849, 2860, 2878, 2884, 2909, 2928, 2933, 2966, 2974, 2988, 3030, 3040, 3066, 3077, 3097, 3103, 3112, 3119, 9, 23, 33, 40, 42, 50, 82, 100, 156, 160, 180, 197, 234, 242, 249, 253, 280, 281, 282, 337, 367, 368, 421, 423, 441, 456, 462, 475, 505, 518, 525, 540, 566, 571, 603, 684, 719, 742, 750, 752, 771, 775, 792, 820, 825, 832, 833, 840, 872, 886, 890, 940, 941, 942, 1023, 1026, 1043, 1052, 1069, 1076, 1114, 1129, 1162, 1173, 1233, 1240, 1249, 1267, 1274, 1310, 1352, 1368, 1410, 1432, 1437, 1443, 1451, 1543, 1584, 1592, 1637, 1642, 1657, 1724, 1767, 1784, 1807, 1865, 1869, 1968, 2002, 2008, 2020, 2021, 2045, 2098, 2115, 2152, 2170, 2174, 2191, 2195, 2255, 2314, 2334, 2340, 2343, 2383, 2393, 2408, 2439, 2471, 2483, 2490, 2531, 2542, 2548, 2625, 2690, 2733, 2736, 2737, 2750, 2758, 2842, 2852, 2897, 2907, 2920, 2921, 2937, 2940, 2954, 2970, 2978, 2981, 3001, 3016, 3020, 3060, 3086, 3088, 3106, 3133, 1373]
    reddit_filtered = [251, 283, 446, 497, 521, 532, 578, 587, 593, 596, 624, 627, 631, 682, 683, 748, 756, 784, 797, 803, 874, 877, 889, 897, 899, 910, 922, 931, 944, 972, 1009, 1020, 1033, 1116, 1123, 1128, 1136, 1139, 1157, 1158, 1168, 1241, 1248, 1256, 1265, 1268, 1273, 1280, 1287, 1288, 1290, 1302, 1313, 1320, 1325, 1335, 1344, 1350, 1354, 1356, 1361, 1364, 1403, 1404, 1412, 1415, 1427, 1440, 1462, 1472, 1477, 1481, 1491, 1503, 1508, 1515, 1517, 1534, 1549, 1557, 1560, 1562, 1566, 1571, 1572, 1609, 1611, 1615, 1618, 1627, 1652, 1660, 1666, 1684, 1695, 1704, 1710, 1711, 1741, 1749, 1755, 1766, 1782, 1786, 1787, 1799, 1802, 1810, 1822, 1824, 1838, 1850, 1851, 1875, 1884, 1890, 1892, 1896, 1898, 1908, 1926, 1944, 1951, 1964, 1997, 2010, 2013, 2034, 2038, 2040, 2055, 2062, 2064, 2065, 2066, 2082, 2090, 2092, 2093, 2111, 2118, 2128, 2130, 2133, 2138, 2142, 2145, 2148, 2151, 2159, 2177, 2194, 2199, 2201, 2206, 2211, 2215, 2230, 2238, 2245, 2274, 2284, 2286, 2293, 2307, 2324, 2331, 2335, 2338, 2339, 2345, 2348, 2350, 2361, 2367, 2368, 2370, 2371, 2398, 2400, 2409, 2420, 2421, 2431, 2435, 2437, 2440, 2441, 2454, 2456, 2462, 2474, 2479, 2484, 2487, 2493, 2498, 2499, 2511, 2512, 2513, 2523, 2525, 2559, 2571, 2576, 2579, 2587, 2590, 2591, 2609, 2611, 2612, 2617, 2620, 2627, 2638, 2642, 2652, 2655, 2662, 2665, 2668, 2669, 2684, 2693, 2703, 2704, 2716, 2718, 2720, 2726, 2731, 2739, 2743, 2744, 2751, 2760, 2788, 2789, 2791, 2793, 2795, 2797, 2800, 2802, 2809, 2814, 2820, 2835, 2855, 2857, 2863, 2870, 2874, 2875, 2882, 2885, 2893, 2896, 2901, 2912, 2916, 2917, 2918, 2924, 2926, 2939, 2941, 2943, 2944, 2945, 2946, 2951, 2957, 2967, 2973, 2982, 2989, 2997, 2999, 3002, 3010, 3013, 3014, 3018, 3029, 3034, 3043, 3046, 3048, 3056, 3058, 3070, 3074, 3079, 3085, 3114, 3125, 3129, 3132, 3134, 3135, 548, 667, 704, 721, 740, 860, 904, 953, 1013, 1045, 1058, 1071, 1207, 1210, 1261, 1284, 1332, 1346, 1482, 1494, 1523, 1617, 1646, 1762, 1775, 1367, 1930, 1938, 1966, 2063, 2070, 2218, 2313, 2707, 2725, 2790, 2845, 2892, 3005, 69, 148, 196, 211, 346, 919, 2500, 3069, 10, 145, 185, 221, 229, 370, 418, 917, 932, 935,  1291, 1511, 1771, 1907, 1972, 2927, 2968, 3082]
    fewshot_given = [0]

    filter_indices = set(onion_filtered + reddit_filtered + fewshot_given)

    # Filter the CSV data
    csv_df_filtered = csv_df[~csv_df['index'].isin(filter_indices)]

    # Merge the filtered CSV and JSON data on the title
    merged_df = pd.merge(json_df, csv_df_filtered, on='title')
    merged_df['response'] = merged_df['response'].apply(lambda x: 1 if x == 1 else 0)

    # Add content length column
    merged_df['content_length'] = merged_df['content'].apply(len)

    # Display statistics for content length
    content_length_stats = merged_df['content_length'].describe()


    print(len(json_df), len(csv_df))
    print(len(merged_df))


    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(merged_df['label'], merged_df['response'])
    precision = precision_score(merged_df['label'], merged_df['response'], zero_division=1)
    recall = recall_score(merged_df['label'], merged_df['response'], zero_division=1)

    # Display the results
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

    print(results)

    y_true = merged_df['label']
    y_pred = merged_df['response']

    target_names = ['Onion', 'Reddit']

    # calculate AUC score

    roc_auc = roc_auc_score(y_true, y_pred)

    # print
    print()
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print(f'The AUC score of the {mode} prompt: {roc_auc}')
    
    return merged_df
    


# Add quantile column based on content length
def calculate_quantiles(df, label):
    label_df = df[df['label'] == label]
    quantiles = label_df['content'].apply(len).quantile([0.2, 0.4, 0.6, 0.8])
    return quantiles

def get_label_quantile(text_length, quantiles):
    if text_length <= quantiles[0.2]:
        return 'q1'
    elif text_length <= quantiles[0.4]:
        return 'q2'
    elif text_length <= quantiles[0.6]:
        return 'q3'
    elif text_length <= quantiles[0.8]:
        return 'q4'
    else:
        return 'q5'

def get_quantile_acc(merged_df):

    quantiles_0 = calculate_quantiles(merged_df, 0)
    quantiles_1 = calculate_quantiles(merged_df, 1)


    merged_df['quantile'] = merged_df.apply(
        lambda x: get_label_quantile(len(x['content']), quantiles_0) if x['label'] == 0 else get_label_quantile(len(x['content']), quantiles_1), 
        axis=1
    )


    q_res = {}

    for quantile in ['q1', 'q2', 'q3', 'q4', 'q5']:
        quantile_df = merged_df[merged_df['quantile'] == quantile]
        results_by_label = {}
        
        for label_value in [0, 1]: # Onion: 0, Reddit: 1
            label_df = quantile_df[quantile_df['label'] == label_value]
            if not label_df.empty:
                accuracy = accuracy_score(label_df['label'], label_df['response'])
                count = len(label_df)
                results_by_label[f'accuracy_label_{label_value}'] = accuracy
                results_by_label[f'count_label_{label_value}'] = count
            else:
                results_by_label[f'accuracy_label_{label_value}'] = None
                results_by_label[f'count_label_{label_value}'] = 0

        q_res[quantile] = results_by_label


    print("Quantile-based accuracy by label:")
    print(q_res)    



def main():
    _ = calculate_acc('zero', zero_res)
    _ = calculate_acc('cot', cot_res)
    merged_df = calculate_acc('few', few_res)
    get_quantile_acc(merged_df)

if name == '__main__':
    main()