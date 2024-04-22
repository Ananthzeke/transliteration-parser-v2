import pandas as pd
import os
import csv,json
from numerize.numerize import numerize

def compute_difference_and_save(json_path, csv_path, output_file_path):
    # Read the CSV files into dataframes
    dct=json.load(open(json_path))
    words=[row[0] for row in csv.reader(open(csv_path, newline=''))][1:]
    print(f'No of words in the current dictionary {numerize(len(dct.keys()),3)}\n')
    print(f'No of words in the current csv file is {numerize(len(words),3)}\n')

    unq_words={"words":list(set(words)-set(dct.keys()))}

    os.makedirs(os.path.dirname(output_file_path),exist_ok=True)

    # Save unq_words to a CSV file
    df=pd.DataFrame(unq_words)
    
    print(f'No of unique words that need to be transliterated {numerize(int(df.size),3)}\n')
    df.to_csv(output_file_path,index=False)

    print(f"Output saved to {output_file_path}")

# Example usage
if __name__=='__main__':
    compute_difference_and_save(
        "san_Deva.json", 
        "unique_words/san_Deva.csv", 
        'new_unq_words/san_Deva.csv')
