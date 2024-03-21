import pandas as pd
import json

def csv2json(path_df,path_json):
    df = pd.read_csv(path_df)
    list=[]
    for i in range(len(df)):
        dict={}
        dict['input']=df['input'][i]
        dict['output']=df['output'][i]
        dict['instruction']=df['instruction'][i]
        list.append(dict)
    #save to json
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(list, f, ensure_ascii=False, indent=4)

def create_instruction(path_df):
    df = pd.read_csv(path_df)
    df['instruction'] = 'สร้างประโยคโฆษณาโดยใช้คำที่กำหนดให้'
    df.to_csv(path_df,index=False)