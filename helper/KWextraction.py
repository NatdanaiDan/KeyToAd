from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
import pandas as pd
from multiprocessing import Pool, cpu_count

def process_row(args):
    index, row = args
    text = row['output']
    list_engine = ["newmm", "longest", "mm", "attacut", "deepcut","nercut"]
    all_words = []

    for i in list_engine:
        word_list = word_tokenize(text, engine=i, join_broken_num=True, keep_whitespace=False)
        pos_tags = pos_tag(word_list)

        noun = [word for word, tag in pos_tags if tag in ['NCMN']]
        adj = [word for word, tag in pos_tags if tag in ['ADVN', 'ADVI', 'ADVP', 'ADVS', 'ADJ']]

        # Collect all words from different engines
        all_words.extend(noun + adj)

    # Identify and return duplicate words across different engines
    duplicate_words = [word for word in set(all_words) if all_words.count(word) > 2]

    return ",".join(duplicate_words)

if __name__ == '__main__':
    path="cosmetic_kwgpt4.csv"
    df = pd.read_csv(path, encoding='utf8')
    list_input=[]
    for i in df.iterrows():
        list_input.append(process_row(i))
    df['input']=list_input
    df.to_csv("cosmetic_kwgpt4.csv",index=False,encoding='utf8')