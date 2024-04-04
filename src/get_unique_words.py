import re,os
import argparse
from glob import glob
from datasets import load_dataset,Dataset
from numerize.numerize import numerize

english_pattern=re.compile(r'[A-Za-z]+')
punct_no_pattern = re.compile(r'[0-9!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~\n\t।|॥۔؟]')


def get_words(ds,column):

    """
    Extract a unique set of words from a specified column in a dataset  by batchwise.

    Args:
        ds: The dataset containing the text data.
        column: The column name in the dataset from which to extract words.

    Returns:
        A list of unique words from the specified column.
    """
    words_set = set()
    for text in ds[column]:
        # removing english words
        text=english_pattern.sub(' ',text)
        # removing punctuation,symbols and numbers 
        text=punct_no_pattern.sub(' ',text).strip()
        words_set.update(word for word in text.split() if word)
    return list(words_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get uniue words from a dataset Hugging  store results in CSV.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input arrow file')
    parser.add_argument('--file_type', type=str,choices=['arrow','csv','parquet','json'] ,default='arrow', help='dataset file type')
    parser.add_argument('--src_lang', type=str, required=True, help='src_lang')
    parser.add_argument('--column_name', type=str, required=True, help='column_name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--cache_dir', type=str, default='/data/umashankar/.cache', help='Cache directory for Hugging Face datasets')
    parser.add_argument('--output_csv_path', type=str, default='output.csv', help='Path to store the output csv file')
    parser.add_argument('--num_proc', type=int, default=8, help='Batch size for processing')
    args = parser.parse_args()

    os.makedirs(args.output_csv_path,exist_ok=True)
    
    ds_path=glob(args.input_path)
    file_type=args.file_type
    cache_dir=args.cache_dir
    num_proc=args.num_proc
    column=args.column_name
    batch_size=args.batch_size
    src_lang=args.src_lang
    output_path=args.output_csv_path

    ds=load_dataset(
        file_type,
        data_files=ds_path,
        cache_dir=cache_dir,
        num_proc=num_proc
        )


    words_ds=ds['train'].map(
        lambda x: {'words':get_words(x,column)},
        batch_size=batch_size,  
        num_proc=num_proc,
        remove_columns=ds.column_names,
        batched=True,
        desc=f'{numerize(ds.num_rows,3)} words'
    )

    #Getting unique words from the dataset
    words_ds=Dataset.from_dict({'words':words_ds.unique('words')})

    print(f'After processed there are  {numerize(words_ds.num_rows,3)} unique words in the language {src_lang}\n')
    
    words_ds.to_csv(f'{output_path}/{src_lang}.csv')
    print(f'Saved the unique words in the path {output_path} for the language {src_lang}')
    
