from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from  indicnlp.tokenize.indic_tokenize  import trivial_tokenize

src_langs=["hin_Deva", "tam_Taml", "asm_Beng", "ben_Beng", "kan_Knda", "mar_Deva", "mal_Mlym", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "tel_Telu", "urd_Arab", "guj_Gujr"]
lang_codes=['hi','ta','as','bn','kn','mr','ml','ne','or','pa','sa','te','ur','gu']
mapping_dict=dict(zip(src_langs,lang_codes))


def normalize(src_lang,text):
    if src_lang not in src_langs:
        raise ValueError(f'Language {src_lang} not supported')
    elif src_lang=='urd_Arab':
        return text
    else:
        Normalizer=IndicNormalizerFactory().get_normalizer(language=mapping_dict[src_lang])
        return Normalizer.normalize(text)
    
if __name__=='__main__':

    text='முனிச்சிலிருந்து ரோம் oவரை நீங்கள் பயணிக்க பல வழிகள் இkங்கேஃ!'
    print(normalize('urd_Arab',text))
    # print(trivial_tokenize(text,lang='ta'))