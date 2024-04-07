import re
import regex
from flashtext import KeywordProcessor
from normalizer import normalize,indic_script_patterns

class MemoryWordReplacer:
    def __init__(self,dictionary_path:str,src_lang:str)->None:
        self.kw_processor=KeywordProcessor()
        self.dictionary=self.kw_processor.add_keyword_from_file(dictionary_path)
        self.src_lang=src_lang
        self.script_suffix=src_lang.split('_')[-1]
        self.compiled_patterns()
        self.load_script_patterns()

    def compiled_patterns(self)->None:
        '''
        Compiles regular expressions used for text processing and assigns them to instance attributes.

        This method initializes various regular expression patterns that are used throughout
        the class for different text processing tasks, such as identifying English words,
        non-romanized characters, and punctuation.

        The method does not return anything but sets the following instance attributes:
        - english_pattern: Pattern to match English alphabetic characters.
        - non_romanized_pattern: Pattern to match non-romanized (Indic) script characters.
        - mixed_word_pattern: Pattern to identify words with a mix of English and Indic characters.
        - nos_and_punctuation_pattern: Pattern to match numbers and punctuation.
        - remove_punctuations_and_symbols: Pattern to remove punctuations and symbols from text.

        Returns:
            None
        '''
        self.english_pattern=re.compile(r'[A-Za-z+]')
        self.non_romanized_pattern = regex.compile(r'[\p{Z}\p{P}\p{S}\p{N}]+')
        self.mixed_word_pattern = re.compile(r'[A-Za-z]')
        self.nos_and_punctuation_pattern=re.compile(r'[\d\.,;:!?\(\)\[\]\{\}\'\"<>@#$%^&*\-_+=/\\|~`]')
        self.remove_punctuations_and_symbols= re.compile(r'^[\s,.!?-]+|[\s,.!?-]+$')

    def load_script_patterns(self)-> None:
        """
        Loads script-specific patterns and compiles a regular expression for text processing.

        This method sets the script patterns based on the source language (src_lang) and 
        compiles a regular expression used for replacing text. It relies on the `indic_script_patterns`
        dictionary and the `script_suffix` derived from the `src_lang` attribute to determine the 
        appropriate pattern for the source language script.

        The method updates the following instance attributes:
        - src_lang_pattern: A regex pattern corresponding to the script of the source language.
        - regex_replacer: A compiled regex object for replacing text based on language-specific rules.

        Raises:
            ValueError: If the script name derived from `src_lang` is not supported in `indic_script_patterns`.

        Returns:
            None
        """
        self.indic_script_patterns = indic_script_patterns
        if self.script_suffix in self.indic_script_patterns:
            self.src_lang_pattern = self.indic_script_patterns[self.script_suffix].pattern
            # It is  a compiled regex pattern specific to the indic language which replaces the whole word. 
            self.regex_replacer = re.compile(
                fr"(?<!{self.src_lang_pattern})({'|'.join(re.escape(key) for key in self.dictionary.keys())})(?!{self.src_lang_pattern})"
            )
        else:
            raise ValueError(f"Script name '{self.script_suffix}' is not supported.")

            
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
    

    
    def mixed_words(self,text:str)->set:
        """
        Identifies words with characters from both English and the target script.

        Args:
            text (str): The text to analyze.

        Returns:
            set: A set of words containing both English and non-English characters.

        Raises:
            ValueError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        return {
            word for word in  self.extract_script_words(text) 
            if bool(self.english_pattern.search(word)) 
        }
    
    
    def fix_mixed_words(self, org_text:str, transliterated_text:str)->str:
        """
        Fix mixed-script words in transliterated text using original text.

        Args:
            org_text (str): Original text.
            transliterated_text (str): Transliterated text with potential mixed-script words.

        Returns:
            str: fixed_text
                fixed_text (str): Transliterated text with mixed-script words fixed.
        """
        try:

            org_text_list = re.split(r'\s+', org_text)
            transliterated_text_list = re.split(r'\s+', transliterated_text)

            print(f'transliterated text:{transliterated_text}\n')
            # Generate mappings
            mixed_words = [
                self.remove_punctuations_and_symbols.sub('',word) 
                for word in list(self.mixed_words(transliterated_text))
                ]
            
            word_mapping={
                self.remove_punctuations_and_symbols.sub('',key):self.remove_punctuations_and_symbols.sub('',value) 
                for key, value in zip(transliterated_text_list,org_text_list)
                }
            if mixed_words:
                pattern = re.compile("|".join(map(re.escape, mixed_words)))
                transliterated_text = pattern.sub(lambda m: word_mapping[m.group()], transliterated_text)

            # Process non-romanized words
            non_romanized_words = self.extract_script_words(self.nos_and_punctuation_pattern.sub(" ",org_text))
            dictionary_lookup = {word: self.dictionary[word] for word in non_romanized_words if word in self.dictionary}
            for word, replacement in dictionary_lookup.items():
                whole_words_pattern=fr"(?<!{self.src_lang_pattern}){word}(?!{self.src_lang_pattern})"
                transliterated_text=re.sub(whole_words_pattern,replacement,transliterated_text)
            return transliterated_text
            
        except Exception as e:

            print(f"Error occurred in fixing_mixed_words: {e}")
            return transliterated_text


    def multiple_replace(self, text:str)->str:
        """
        Replace words in the given text using a dictionary and handle language-specific replacements.

        Args:
            text (str): Input text to perform replacements on.

        Returns:
            str: Text with words replaced according to the dictionary and language-specific rules.

        """

        # Normalize text before replacements 
        text=f" {normalize(src_lang=self.src_lang,text=text)} "
            
        try:
            #trying flashtext replacements
            text=self.kw_processor.replace_keywords(text)

        except Exception as e:
            #if it fails then use regex based replacements
            text=self.regex_replacer.sub(lambda x: self.dictionary[x.group()], text)
        return text.strip()


    
    def replace_batches(self, batch:list, use_placeholder:bool=True)->tuple[list,list]:

        """
        Processes a batch of text lines, replacing words based on the dictionary and handling mixed scripts.

        Args:
            batch (list): List of text lines to process.
            use_placeholder (bool): Flag to use a placeholder for batch processing.
            The default identification of for a batch is [batch]

        Returns:
            tuple: A tuple containing the processed text batch and a list of missing words.
        """

        if not self.dictionary:
            print('Dictionary is None')
            return batch, [[''] * len(batch)]

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
            return batch, [batch.split()]
        fixed_batch = [
            self.fix_mixed_words(org_string, transliterated_string)
            for org_string, transliterated_string in zip(batch, transliterated_batch)
        ]
        # Missing words extraction and handling

        missing_words = [self.extract_script_words(sent) or [''] for sent in fixed_batch]

        if len(batch) != len(fixed_batch):
            fixed_batch = self.replace_batches(batch, use_placeholder=False)

        return fixed_batch, missing_words

    
if __name__=='__main__':

    a=MemoryWordReplacer('dictionaries/Final_Dict/tam_Taml_final.json','tam_Taml')
    text=[
        'இந்தமூன்று விருப்பங்களும் முனி 55முனிச்சிலிருந்து ரோம்-வரை பயணிக்க சாத்தியமான வழிகளாக இருக்கலாம் மூன்றுகவிருப்பங்களும் .',
          'வழிகளாக வழிகளாக'
          ]
    text=['''".
        3.	1972ஆம் .
        4.	1707இல் .
        5.	1700களில்
        "''']
    print(a.replace_batches(text))
