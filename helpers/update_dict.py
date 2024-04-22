import json,os
from numerize.numerize import numerize


def combine_json_files(input_paths, output_path):
    combined_dict = {}
    dir=os.path.dirname(output_path)
    os.makedirs(dir,exist_ok=True)
    # Read each JSON file and add its content to the combined dictionary
    for path in input_paths:
        with open(path, 'r') as file:
            data = json.load(file)
            combined_dict.update(data)
    print(f'No of words in a combined dictionary is {numerize(len(combined_dict.keys()),3)}')
    # Save the combined dictionary to a new JSON file
    with open(output_path, 'w') as file:
        json.dump(combined_dict, file, indent=4,ensure_ascii=False)
    print(f'sucessfully saved in {output_path}')

if __name__=='__main__':
    lang='urd_Arab'
    path_1=f'/dictionary/{lang}.json'
    path_2=f'/dictionary_2/{lang}.json'
    # input_paths=glob(f'{lang}/*chunk*/*json')
    input_paths=[path_1,path_2]
    print(input_paths)
    output_path=f'universal_dictionary/{lang}.json'
    combine_json_files(input_paths,output_path)