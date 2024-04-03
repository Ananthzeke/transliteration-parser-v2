from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import re

# IndicTrans lang code
src_langs=["hin_Deva", "tam_Taml", "asm_Beng", "ben_Beng", "kan_Knda", "mar_Deva", "mal_Mlym", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "tel_Telu", "urd_Arab", "guj_Gujr"]

# Indicnlp lang code
lang_codes=['hi','ta','as','bn','kn','mr','ml','ne','or','pa','sa','te','ur','gu']
mapping_dict=dict(zip(src_langs,lang_codes))

#regex complied  Indic script patterns
indic_script_patterns={
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


thandaa_pattern = re.compile(r'[।|॥]')

def normalize(src_lang:str ,text:str )-> str:
    """
    Normalize text in specified South Asian language script.
    
    Args:
        src_lang (str): Source language code.
        text (str): Text to normalize.
    
    Returns:
        str: Normalized text.
    
    Raises:
        ValueError: If `src_lang` is not supported.
    """
    if not isinstance(str,text):
        raise ValueError(f'Normalizer only supports string as an input')

    if src_lang not in src_langs:
        raise ValueError(f'Language {src_lang} not supported')
    
    if src_lang.split('_')[-1] in ['Orya', 'Deva', 'Beng', 'Gujr', 'Guru']:
        text=thandaa_pattern.sub('.',text)

    elif src_lang=='urd_Arab':
        text=text.replace('۔', '.').replace('؟', '?')
        return text
    else:
        Normalizer=IndicNormalizerFactory().get_normalizer(language=mapping_dict[src_lang])
        return Normalizer.normalize(text)
    
if __name__=='__main__':

    text='முனிச்சிலிருந்து ரோம் oவரை நீங்கள் பயணிக்க பல வழிகள் இkங்கேஃ!'
    print(normalize('tam_Taml',text))
    # print(trivial_tokenize(text,lang='ta'))