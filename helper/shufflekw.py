import json
import random

def shuffle_and_drop_keywords(before_json_path, after_json_path,num_augment=2):
    with open(before_json_path) as json_file:
        data = json.load(json_file)

    new_data = []

    def shuffle_and_drop(item):
        words = item['input'].split(',')
        random.shuffle(words)

        if len(words) < 4:
            min_len = 1
            max_len = len(words)
        elif len(words) <= 8:
            min_len = 2
            max_len = len(words)//2
        else:
            min_len = 3
            max_len = len(words)//2
        # print(min_len, max_len)
        num_to_keep = random.randint(min_len,max_len)
        text = ','.join(words[:num_to_keep])
        new_dict = {'input': text, 'output': item['output'], 'instruction': item['instruction']}
        new_data.append(new_dict)

    # Apply shuffle_and_drop function to each item in data
    for item in range(len(data)):
        for i in range(num_augment):
            shuffle_and_drop(data[item])

    # Merge data
    print("Original data length:", len(data))
    print("Augmented data length:", len(new_data))
    print("Total data length:", len(data) + len(new_data))
    new_data = new_data + data
    #drop duplicate
    

    with open(after_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_data, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    before_json_path = '/home/natdanai/KeyToad/keywordjson/furniture_kw_clean.json'
    after_json_path = 'furniture_kw_augmented.json'
    shuffle_and_drop_keywords(before_json_path, after_json_path)
