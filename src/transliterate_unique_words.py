import os,re
import json
import argparse
from numerize.numerize import numerize
from datasets import load_dataset,Dataset,concatenate_datasets
from ai4bharat.transliteration import XlitEngine
from normalizer import mapping_dict,indic_script_patterns


english_pattern=re.compile(r'[A-Za-z]+')
punct_no_pattern = re.compile(r'[0-9!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~\n\t।|॥۔؟]')
punct_no_pattern_in_mid = re.compile(r'\w[ !-\/:-@\[-`{-~\d]\w')

#Intialize indicxlit engine
engine =  XlitEngine( beam_width=4, src_script_type = "indic")


def contains_space_symbol_or_number_in_middle(word):
    """
    Check if a word has a space, symbol, or number in the middle.
    """
    return bool(punct_no_pattern_in_mid.search(word))

def remove_punctuation_and_numbers(batch_words):
    """
    Clean a list of words by removing punctuation and numbers.
    """
    batch_words=[w for word in batch_words for w in punct_no_pattern.sub(' ',word).strip().split(' ') if w]
    return batch_words


def contains_punctuation(word):
    """
    Determine if a word contains punctuation.
    """
    return bool(punct_no_pattern.search(word))

def contains_english_words(s):
    """
    Check if a string contains English letters.
    """
    return bool(english_pattern.search(s))


def ds_to_json(ds,column):
    # Convert to Pandas DataFrame
    df = ds.to_pandas()
    
    # Ensure the columns are named correctly
    if column not in df.columns or 'transliterated' not in df.columns:
        raise ValueError(f"Expected columns {column} and 'transliterated' not found")

    # Convert the DataFrame to a dictionary
    dictionary = df.set_index(column)['transliterated'].to_dict()
    return dictionary



def transliterate(org_batch,src_lang,use_sentence_transliterate=False):
    """
    Transliterates a batch of text from the source language to English.

    This function supports both word-level and sentence-level transliteration.

    Parameters:
    org_batch (list of str): The original text batch to be transliterated.
    src_lang (str): The source language code (e.g., 'hi', 'ta').
    use_sentence_transliterate (bool): Flag to use sentence-level transliteration if True; otherwise, word-level.

    Returns:
    dict: A dictionary with a key 'transliterated' containing the list of transliterated text.
    
    Note:sentence transliterate preserves symbols,punctuations, numbers and other languages

    """
    if use_sentence_transliterate:
         # Combine the batch into a single string for sentence-level transliteration
        batch='[batch]'.join(org_batch)
        try:
            batch=engine._transliterate_sentence(text=batch,src_lang=src_lang,tgt_lang='en').split('[batch]')
            # Verify if transliteration count matches input count
            if len(batch)!=len(org_batch):
                batch=[engine._transliterate_sentence(text=word,src_lang=src_lang,tgt_lang='en') for word in org_batch]

        except Exception as e:
            print(f'Failed on sentence transliteration due to {e.message}')

        return {'transliterated':batch}
    
    else:
        try:
            batch=engine.batch_transliterate_words(
                    org_batch,
                    src_lang=src_lang,
                    tgt_lang='en',
                    topk=1
            )
            #check the length of the batch
            assert len(org_batch)==len(batch[0])

        except Exception as e:
            print(f'Failed on batch transliteration due to {e if e else "input size not equal to output size"}, continuing with word transliteration')
            
            # Word by word transliteration
            batch=[[engine.translit_word(word,src_lang,topk=1)[0] for word in org_batch]]

        return {'transliterated':batch[0]}        


def transliterate_using_hugging_face(input_path,column,src_lang,batch_size,cache_dir,num_proc=8):
    
    ds=load_dataset(
        'csv',
        data_files=input_path,
        cache_dir=cache_dir
    )

    #de-dup
    ds=ds['train'].to_pandas()
    ds = ds[ds[column].notnull()].drop_duplicates(column).reset_index()
    ds=Dataset.from_pandas(ds)
    sent_ds=ds.filter(
        lambda x : contains_space_symbol_or_number_in_middle(x[column])
         or 
         contains_english_words(x[column]),num_proc=num_proc
         )
    sent_ds=sent_ds.to_pandas().drop_duplicates(column)
    sent_ds=Dataset.from_pandas(sent_ds)
    ds=ds.filter(
        lambda x : not contains_space_symbol_or_number_in_middle(x[column]) 
        and not contains_english_words(x[column]) 
        and len(x[column])<=100,num_proc=num_proc
        )
    
    ds=ds.map(lambda batch : {
        column:remove_punctuation_and_numbers(batch[column])
        },
        remove_columns=ds.column_names,
        num_proc=num_proc,
        batched=True)
    script=src_lang.split('_')[-1]
    ds=ds.filter(lambda x : indic_script_patterns[script].search(x[column]),num_proc=num_proc)
    ds=ds.to_pandas().drop_duplicates(column)
    ds=Dataset.from_pandas(ds)

    if ds.num_rows:
        ds=ds.map(
            lambda x: transliterate(x[column],mapping_dict[src_lang],False),
            batched=True,
            batch_size=batch_size,
            desc=f'batch transliteration ({numerize(ds.num_rows,3)} words)'
    
        )
    if sent_ds.num_rows:
        sent_ds=sent_ds.map(
            lambda x: transliterate(x[column],mapping_dict[src_lang],True),
            batched=True,
            batch_size=batch_size,
            desc=f'sentence transliteration ({numerize(sent_ds.num_rows,3)} words)'
        )
    ds=concatenate_datasets([ds,sent_ds])
    print(f'\nTotal words transliterated {numerize(ds.num_rows,3)}')

    return ds

def store_data_as_json(input_data,src_lang,file_path):
    """
    Store the given data as a JSON file.
    
    Args:
    input_data (dict): The data to be stored.
    file_path (str): The path where the JSON file will be saved.
    
    Returns:
    None: The function doesn't return anything but saves the data to a file.

    """
    os.makedirs(file_path,exist_ok=True)
    with open(f'{file_path}/{src_lang}.json', 'w') as file:
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
    parser.add_argument('--num_proc', type=int, default=8, help='Batch size for processing')
    args = parser.parse_args()

    # Use the parsed arguments
    ds = transliterate_using_hugging_face(
        args.input_path,
        args.column_name,
        args.src_lang,
        args.batch_size,
        args.cache_dir,
        args.num_proc
    )

    # Save the dataset to JSON
    ds_dict = ds_to_json(ds,args.column_name)
    store_data_as_json(ds_dict, args.src_lang,args.output_json_path)