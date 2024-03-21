import os
import glob
import argparse
from MemoryWordReplacer import MemoryWordReplacer
from datasets import load_dataset,disable_caching,Features,Sequence,Value

disable_caching()



def create_dir_if_not_exists(file_path):
    """Checks if the directory containing a file exists and creates it if it doesn't.

    Args:
        file_path (str): The path to the file, including its directory.
    """
    directory = os.path.dirname(file_path)  # Extract the directory path from the file path
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset for transliteration.')
    parser.add_argument('--dictionary_path', type=str, required=True, help='Path to the dictionary JSON file.')
    parser.add_argument('--src_lang', type=str, required=False, help='Source language of the text')
    parser.add_argument('--cache_dir', type=str, default=None,required=True, help='Cache directory for storing temporary files.')
    parser.add_argument('--id_column', type=str, default='doc_id', help='Column to be processed.')
    parser.add_argument('--text_column', type=str, default='translated', help='Column to be processed.')
    parser.add_argument('--file_type', type=str, required=True,choices=['csv','parquet','arrow'])
    parser.add_argument('--dataset_path', type=str, required=True, help='Glob path for dataset files.')
    parser.add_argument('--missing_log_path', type=str, required=True, help='Name for missing words text file')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of processes to use.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing.')
    parser.add_argument('--sample_size', type=int, help='Sample size to select from dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the processed dataset.')

    args = parser.parse_args()


    dictionary_path=args.dictionary_path
    cache_dir=args.cache_dir
    src_lang=args.src_lang
    id_column=args.id_column
    translated_text_column=args.text_column
    file_type=args.file_type
    dataset_paths=glob.glob(args.dataset_path)
    sample_size=args.sample_size
    batch_size=args.batch_size
    num_proc=args.num_proc
    missing_words_log_path=args.missing_log_path
    output_path=args.output_path

    create_dir_if_not_exists(missing_words_log_path)

    ds=load_dataset(
        file_type,
        data_files=dataset_paths,
        cache_dir=cache_dir,
    ).select_columns([id_column,translated_text_column])

    ds=ds.filter(lambda x : x[translated_text_column] not in (None,''),num_proc=num_proc)
    if sample_size:
        ds = ds['train'].select(range(sample_size))
    else:
        ds=ds['train']

    mem_replacer=MemoryWordReplacer(dictionary_path,src_lang=src_lang)

    out_columns=['transliterated','missing_words']
    out_features=Features({
        id_column:Value("string"),
        translated_text_column:Value("string"),
        out_columns[0]: Value("string"),
        out_columns[1]: Sequence(Value("string"))
        })
    ds=ds.map(
        lambda z:dict(zip(out_columns,mem_replacer.replace_batches(z[translated_text_column]))),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        features=out_features
    )
    df=ds.to_pandas()['missing_words'].explode().drop_duplicates()
    df.to_csv(f'{missing_words_log_path}/{src_lang}.csv',index=False)

    ds.to_csv('sample.csv')
    if ds.num_rows//2>num_proc and num_proc>=40:
        ds.save_to_disk(output_path,num_proc=40)
    else:
        ds.save_to_disk(output_path)