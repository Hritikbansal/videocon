import os
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--input_csv', type = str, required = True, help = 'input csv file')
parser.add_argument('--output_csv', type = str, required = True, help = 'output csv file')
parser.add_argument('--feedback', action = 'store_true', help = 'add feedback data')
parser.add_argument('--entailment', action = 'store_true', help = 'add entailment data')

args = parser.parse_args()

PROMPT_LABEL = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Does this video entail the description: "{caption}"?
AI: {label}'''

PROMPT_FEEDBACK = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: What is the misalignment between this video and the description: "{caption}"?
AI: {feedback}'''

def main():

    df = pd.read_csv(args.input_csv)
    df = df.drop_duplicates()
    print(df.head())
    print(len(df))

    seq_data = defaultdict(list)
    for j in tqdm(range(len(df))):
        videopath = df.iloc[j]['videopath']
        if args.entailment:
            seq_data['videopath'].append(videopath)
            seq_data['videopath'].append(videopath)
            seq_data['caption'].append(PROMPT_LABEL.format(caption = df.iloc[j]['caption'], label = 'Yes'))
            seq_data['caption'].append(PROMPT_LABEL.format(caption = df.iloc[j]['neg_caption'], label = 'No'))
            seq_data['split'].append(df.iloc[j]['split'])
            seq_data['split'].append(df.iloc[j]['split'])
        if args.feedback:
            seq_data['videopath'].append(videopath)
            seq_data['split'].append(df.iloc[j]['split'])
            seq_data['caption'].append(PROMPT_FEEDBACK.format(caption = df.iloc[j]['neg_caption'], feedback = df.iloc[j]['nle']))
    df = pd.DataFrame(seq_data)
    df = df.drop_duplicates()
    df_train = df[df['split'] == 'train']
    df_val   = df[df['split'] == 'val']
    df_test  = df[df['split'] == 'test']
    df_train.to_csv(args.output_csv, index = False)
    df_val.to_csv(args.output_csv.replace('train', 'val'), index = False)
    df_test.to_csv(args.output_csv.replace('train', 'test'), index = False)

if __name__ == "__main__":
    main()