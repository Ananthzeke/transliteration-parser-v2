#!/bin/bash

# Path to your Python script
SCRIPT_PATH="src/transliterate_unique_words.py"

# Arguments for your Python script
INPUT_PATH="tam_Taml.csv"
SRC_LANG="ta"
BATCH_SIZE=64
CACHE_DIR="/data/umashankar/.cache"
OUTPUT_JSON_PATH="tam.json"
COLUMN='missing_words'
# Execute the Python script with the specified arguments
HF_CACHE='/data/umashankar/.cache' python $SCRIPT_PATH --input_path "$INPUT_PATH" --column_name "$COLUMN" --src_lang "$SRC_LANG" --batch_size $BATCH_SIZE --cache_dir "$CACHE_DIR" --output_json_path "$OUTPUT_JSON_PATH"