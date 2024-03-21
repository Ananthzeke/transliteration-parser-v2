import os,re
import json
import glob
import argparse
from datasets import load_dataset,Dataset
from ai4bharat.transliteration import XlitEngine
# from data_prepartion.prefix_heirarchy import ds_to_json
english_pattern=re.compile(r'[A-Za-z]+')

engine =  XlitEngine( beam_width=4, src_script_type = "indic")

def ds_to_json(ds,column):
    # Convert to Pandas DataFrame
    df = ds.to_pandas()
    
    # Ensure the columns are named correctly
    if column not in df.columns or 'transliterated' not in df.columns:
        raise ValueError(f"Expected columns {column} and 'transliterated' not found")

    # Convert the DataFrame to a dictionary
    dictionary = df.set_index(column)['transliterated'].to_dict()
    return dictionary

def remove_english_words(s):
    return english_pattern.sub("", s)

def transliterate(org_batch,src_lang,use_sentence_transliterate=False):
    if use_sentence_transliterate:
        batch=[ engine._transliterate_sentence(text=word,src_lang=src_lang,tgt_lang='en') 
        for word in org_batch]
        return {'transliterated':batch}
    
    else:
        batch=engine.batch_transliterate_words(
                org_batch,
                src_lang=src_lang,
                tgt_lang='en',
                topk=1
            )

        if len(org_batch)!=len(batch[0]):
            batch=[engine.translit_word(word,src_lang,topk=1)[0] for word in org_batch]
            return {'transliterated':batch}
        return {'transliterated':batch[0]}

def transliterate_using_hugging_face(input_path,column,src_lang,batch_size,cache_dir,use_sentence_transliterate):
    
    ds=load_dataset(
        'csv',
        data_files=input_path,
        cache_dir=cache_dir
    )

    #de-dup
    ds=ds.filter(lambda x: True if x[column]!=None   else False )
    ds=ds.map(
        lambda x: {transliterate(x[column],src_lang,use_sentence_transliterate)},
        batched=True,
        batch_size=batch_size,
    )
    print(ds)

    return ds

def store_data_as_json(input_data, file_path):
    """
    Store the given data as a JSON file.
    
    Args:
    input_data (dict): The data to be stored.
    file_path (str): The path where the JSON file will be saved.
    
    Returns:
    None: The function doesn't return anything but saves the data to a file.
    """
    with open(file_path, 'w') as file:
        json.dump(input_data, file, indent=4,ensure_ascii=False)
    print(f'saved dict in {file_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transliterate words using Hugging Face and store results in JSON.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--column_name', type=str, required=True, help='column_name')
    parser.add_argument('--src_lang', type=str, required=True, help='Source language code')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--cache_dir', type=str, default='/data/umashankar/.cache', help='Cache directory for Hugging Face datasets')
    parser.add_argument('--output_json_path', type=str, default='output.json', help='Path to store the output JSON file')
    parser.add_argument('--use_sentence_transliterate', action='store_true', help='sentence transliterate preserves structure of the symbols')
    args = parser.parse_args()

    # Use the parsed arguments
    ds = transliterate_using_hugging_face(
        args.input_path,
        args.column_name,
        args.src_lang,
        args.batch_size,
        args.cache_dir,
        args.use_sentence_transliterate
    )

    # Save the dataset to JSON
    ds_dict = ds_to_json(ds['train'],args.column_name)
    store_data_as_json(ds_dict, args.output_json_path)