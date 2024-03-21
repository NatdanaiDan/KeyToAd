import random
import pandas as pd
import json
from pythainlp.tag import pos_tag

df = pd.read_csv("/home/natdanai/Download/dataset/cleaned Furniture_KW_new_drop.csv")
#get Keyword from column "Keyword"
# Convert the 'Keyword' column from string representation of list to actual list
#print column name df
df['input'] = df['input'].apply(lambda x: x.split(','))

# Create a dictionary with running numbers as keys and keywords as values
keyword_dict = dict(enumerate(df['input']))

def get_random_value():
    while True:
        chosen_value = random.choice(list(keyword_dict.values()))
        if all(value.strip() for value in chosen_value) and len(chosen_value) > 4:
            return chosen_value

def get_random_subset():
    # Get a random list of keywords
    list_keyword = get_random_value()
    
    # Randomly select a number between 3 to 5
    # num_keywords = random.randint(5, 5)
    
    # Randomly select that many keywords from the list
    # if num_keywords <= len(list_keyword):
    #     list_keyword=random.sample(list_keyword, num_keywords)
    # else:
    #     list_keyword=random.sample(list_keyword, len(list_keyword))
    
    return random.sample(list_keyword, random.randint(3,5))



with open('/home/natdanai/KeyToad/keywordjson/Cleaned_keywords.json',encoding='utf-8') as f:
    keywordjson = json.load(f)

def getshfflekeyword():
    #random 3 to 5 form keyword
    shufflekeyword=random.sample(keywordjson,random.randint(5,7))
    return shufflekeyword

def check_NCMN(keywordlist):
    #check NCMN in keywordlist
    return any(pos_tag([keyword])[0][1] == 'NCMN' for keyword in keywordlist)


with open('/home/natdanai/KeyToad/keywordjson/Furnitures_keywords.json',encoding='utf-8') as f:
    furnituresjson = json.load(f)

with open('/home/natdanai/KeyToad/keywordjson/Locations__Mood_keywords.json',encoding='utf-8') as f:
    locationsjson = json.load(f)

def getfunrituresandlocation():
    #random 3 to 5 form keyword
    furnitureskeyword=random.sample(furnituresjson,1)
    locationskeyword=random.sample(locationsjson,1)
    randomkeyword=random.sample(keywordjson,random.randint(2,5))
    #drop duplicate
    shufflekeyword=furnitureskeyword+locationskeyword+randomkeyword
    shufflekeyword = list(set(shufflekeyword))

    return shufflekeyword

if __name__ == "__main__":
    print(getfunrituresandlocation())
    print(','.join(getfunrituresandlocation()))