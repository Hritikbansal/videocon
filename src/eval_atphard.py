import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--input_csv_1', type = str, required = True, help = 'input csv file 1: ground truth')
parser.add_argument('--input_csv_2', type = str, required = True, help = 'input csv file 2: prediction')
parser.add_argument('--input_csv_3', type = str, default = "nextqa-atphard.csv", help = 'original data file')

args = parser.parse_args()

def main():

    df1 = pd.read_csv(args.input_csv_1)
    df2 = pd.read_csv(args.input_csv_2, names = ['videopath', 'caption', 'entailment', 'pos', 'neg'])
    df1 = df1.drop_duplicates()
    df2 = df2.drop_duplicates()
    print(len(df1), len(df2))


    df = pd.merge(df1, df2, on = ['videopath', 'caption'], how = 'inner')
    df = df.drop_duplicates()
    print(len(df))

    map_id_ques_type = {}
    df3 = pd.read_csv(args.input_csv_3)
    for j in range(len(df3)):
        map_id_ques_type[f"{df3.iloc[j]['video']}-{df3.iloc[j]['question']}"] = df3.iloc[j]['type']

    res = defaultdict(int)
    count = 0
    total = 0
    for _, tmp_df in df.groupby(['videopath', 'question']):
        if len(tmp_df) == 5:
            prediction = np.argmax(tmp_df['entailment'].tolist())
            video_id = tmp_df.iloc[0]['videopath']
            video_id = video_id.split("/")[-1][:-4]
            question = tmp_df.iloc[0]['question']
            type_ques = map_id_ques_type[f"{video_id}-{question}"]
            if prediction == tmp_df.iloc[0]['answer']:
                count += 1
                res[f"{type_ques[0]}-count"] += 1
            res[f"{type_ques[0]}-total"] += 1            
            total += 1
    print(total)
    print(f"Accuracy: {100 * count / total}")
    print(f"Accuracy-C: {100 * res['C-count'] / res['C-total']}")
    print(f"Accuracy-T: {100 * res['T-count'] / res['T-total']}")

if __name__ == "__main__":
    main()

'''
    python eval_nextqa.py --input_csv_1 /local2/hbansal/nextqa/eval-nextqa-atphard.csv --input_csv_2 /local2/hbansal/nextqa/mplug_msrvtt_pretrained_nextqa.csv 
    python eval_nextqa.py --input_csv_1 /local2/hbansal/nextqa/ib-eval-nextqa-atphard.csv --input_csv_2 /local2/hbansal/nextqa/imagebind_pretrained.csv 
'''