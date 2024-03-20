tgt_lang='hin_Deva'
python src/main.py \
        --dictionary_path "dictionaries/Final_Dict/${tgt_lang}_final.json" \
        --cache_dir ".cache" \
        --dataset_path "exp.parquet" \
        --text_column "${tgt_lang}" \
        --id_column "doc_id" \
        --num_proc  8\
        --batch_size  10\
        --output_path "transliteration/${tgt_lang}_transliterated" \
        --file_type 'parquet' \
        --missing_log_path "missing_words/"\
        --src_lang "${tgt_lang}"\
        --sample_size 1000
echo "${tgt_lang}" has been finished