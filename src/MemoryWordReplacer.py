import re
import regex
from flashtext import KeywordProcessor
from normalizer import normalize

class MemoryWordReplacer:
    def __init__(self,dictionary_path:str,src_lang:str):
        self.kw_processor=KeywordProcessor()
        self.dictionary=self.kw_processor.add_keyword_from_file(dictionary_path)
        self.english_pattern=re.compile(r'[A-Za-z+]')
        self.non_romanized_pattern = regex.compile(r'[\p{Z}\p{P}\p{S}\p{N}]+')
        self.mixed_word_pattern = re.compile(r'[A-Za-z]')
        self.allowed_mixed_word_pattern = re.compile(r'^[a-zA-Z0-9\s.,\'"!?]+$')
        self.regex_replacer=re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in self.dictionary.keys()) + r')(?!\w)')
        self.nos_and_punctuation_pattern=re.compile(r'[\d\.,;:!?\(\)\[\]\{\}\'\"<>@#$%^&*\-_+=/\\|~`]')
        self.remove_punctuations_and_symbols= re.compile(r'^[\s,.!?-]+|[\s,.!?-]+$')
        self.src_lang=src_lang
        self.script_suffix=src_lang.split('_')[-1]
        self.indic_script_patterns={
        "Arab": re.compile(r"[\u0600-\u06FF]"),
        "Beng": re.compile(r"[\u0980-\u09FF]"),
        "Deva": re.compile(r"[\u0900-\u097F]"),
        "Guru": re.compile(r"[\u0A00-\u0A7F]"),
        "Gujr": re.compile(r"[\u0A80-\u0AFF]"),
        "Orya": re.compile(r"[\u0B00-\u0B7F]"),
        "Taml": re.compile(r"[\u0B80-\u0BFF]"),
        "Telu": re.compile(r"[\u0C00-\u0C7F]"),
        "Knda": re.compile(r"[\u0C80-\u0CFF]"),
        "Mlym": re.compile(r"[\u0D00-\u0D7F]"),
    }

    def remove_english_words(self,s):
        return self.english_pattern.sub('', s)
        
    def remove_english_words_and_empty_strings(self,words):
        pattern = re.compile(r'^[a-zA-Z]+$')
        filtered_words = [word for word in words if not pattern.match(word) and word.strip()]
        return filtered_words

    def split_non_romanized_string(self, text):
            words = [word.strip() for word in self.non_romanized_pattern.split(text) if self.remove_english_words(word)]
            return words
            
    def extract_script_words(self, sentence):
        """
        Extract words from a sentence that belong to a Indic script.

        Args:
            sentence (str): Input sentence to extract words from.

        Returns:
            list: List of words from the sentence that belong to the specified script.
        """
        if self.script_suffix not in self.indic_script_patterns:
            print (f"Script name '{self.script_suffix}' is not supported.")
        pattern = self.indic_script_patterns[self.script_suffix]
        # Split the sentence into words and filter those that contain characters matching the script's pattern
        if sentence!=None:
            words = [word for word in sentence.split() if pattern.search(word)]
            return words
        else:
            return [""]
    

    
    def mixed_words(self,text):
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        return {
            word for word in  self.extract_script_words(text) 
            if bool(self.english_pattern.search(word)) 
        }
    
    
    def fix_mixed_words(self, org_text, transliterated_text):
        """
        Fix mixed-script words in transliterated text using original text.

        Args:
            org_text (str): Original text.
            transliterated_text (str): Transliterated text with potential mixed-script words.

        Returns:
            tuple: (fixed_text, missing_words)
                fixed_text (str): Transliterated text with mixed-script words fixed.
                missing_words (list): Words from original text not found in dictionary.
        """
        try:
            org_text_list=org_text.replace('\n',' ').split(' ')
            transliterated_text_list=transliterated_text.replace('\n',' ').split(' ')
            # Generate mappings
            word_mapping={
                self.remove_punctuations_and_symbols.sub('',key):self.remove_punctuations_and_symbols.sub('',value) 
                for key, value in zip(transliterated_text_list,org_text_list)
                }

            mixed_words = [
                self.remove_punctuations_and_symbols.sub('',word) 
                for word in list(self.mixed_words(transliterated_text))
                ]
            
            if mixed_words:
                    pattern = re.compile("|".join(map(re.escape, mixed_words)))
                    transliterated_text = pattern.sub(lambda m: word_mapping[m.group()], transliterated_text)
            

            # Process non-romanized words
            # print(self.nos_and_punctuation_pattern.sub(" ",org_text))
            non_romanized_words = self.extract_script_words(self.nos_and_punctuation_pattern.sub(" ",org_text))
            dictionary_lookup = {word: self.dictionary[word] for word in non_romanized_words if word in self.dictionary}
            for word, replacement in dictionary_lookup.items():
                print(f'{word}:{replacement}')
                transliterated_text = transliterated_text.replace(f' {word} ', f' {replacement} ')

                return transliterated_text
            
        except Exception as e:
            print(f"Error occurred in fixing_mixed_words: {e}")
            # print(mixed_words)
            # print(word_mapping)
            return transliterated_text


    def multiple_replace(self, text):
        """
        Replace words in the given text using a dictionary and handle language-specific replacements.

        Args:
            text (str): Input text to perform replacements on.

        Returns:
            str: Text with words replaced according to the dictionary and language-specific rules.

        """
        text=f" {normalize(src_lang=self.src_lang,text=text)} "
            
        try:
            text=self.kw_processor.replace_keywords(text)
            
        except Exception as e:
            print(e,'trying regex replacer')
            text=self.regex_replacer.sub(lambda x: self.dictionary[x.group()], text)

        return text.strip()
    
    def replace_numbers_and_punctuation(self,text):        
        # Replace all occurrences of the pattern with a space
        return self.nos_and_punctuation_pattern.sub(' ', text).strip().split(' ')


    
    def replace_batches(self, batch, use_placeholder=True):
        if not self.dictionary:
            print('Dictionary is None')
            return batch, [[''] * len(batch)]

        # Compute language-specific operations once
        
        # Replacing punctuations
        if self.script_suffix in ['Orya', 'Deva', 'Beng', 'Gujr', 'Guru']:
            pattern = re.compile(r'[।|॥]')
            batch = [pattern.sub('.', text) for text in batch]
        elif self.script_suffix == 'Arab':
            batch = [text.replace('۔', '.').replace('؟', '?') for text in batch]

        # Placeholder handling
        if use_placeholder:
            placeholder = '[batch]'
            text = placeholder.join(batch)
            transliterated_text = self.multiple_replace(text)
            transliterated_batch = transliterated_text.split(placeholder)
        else:
            transliterated_batch = [self.multiple_replace(text) for text in batch]

        if not transliterated_batch:
            print('Failed on transliteration returning Original text')
            return batch, [[''] * len(batch)]

        fixed_batch = [
            self.fix_mixed_words(org_string, transliterated_string)
            for org_string, transliterated_string in zip(batch, transliterated_batch)
        ]

        # Missing words extraction and handling
        missing_words = [self.extract_script_words(self.nos_and_punctuation_pattern.sub(" ",sent)) for sent in fixed_batch]
        if missing_words:
            all_missing_words = set(sum(missing_words, []))
            for word in all_missing_words:
                if word in self.dictionary:
                    fixed_batch = [sent.replace(word, self.dictionary[word]) if sent else sent for sent in fixed_batch]

        missing_words_new = [self.extract_script_words(sent) or [''] for sent in fixed_batch]

        if len(batch) != len(fixed_batch):
            fixed_batch = self.replace_batches(batch, use_placeholder=False)

        return fixed_batch, missing_words_new

    
if __name__=='__main__':

    a=MemoryWordReplacer('dictionaries/Final_Dict/tam_Taml_final.json','tam_Taml')
    text=['இந்த மூன்று விருப்பங்களும் முனி 55முனி:ச்சிலிருந்து ரோம் வரை பயணிக்க சாத்தியமான வழிகளாக இருக்கலாம்.']
    print(a.extract_script_words(text[0]))
    new_text=a.replace_batches(text)
    print(new_text)
