import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--input_csv', type = str, required = True, help = 'input csv file')
parser.add_argument('--video_dir', type = str, required = True, help = 'video directory')
parser.add_argument('--map_json', type = str, required = True, help = 'video mapping')
parser.add_argument('--output_csv', type = str, help = 'output csv file')

args = parser.parse_args()

PROMPT = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Does this video entail the description: "{caption}"?
AI: '''

def main():
    
    df = pd.read_csv(args.input_csv)
    
    with open(args.map_json, 'r') as f:
        mapping = json.load(f)

    res = defaultdict(list)
    for j in tqdm(range(len(df))):
        res['videopath'] = res['videopath'] + (5 * [os.path.join(args.video_dir, f"{mapping[str(df.iloc[j]['vid_id'])]}.mp4")])
        res['question']  = res['question'] + (5 * [df.iloc[j]['question']])
        res['answer']    = res['answer'] + (5 * [df.iloc[j]['answer']])
        res['option']  = res['option']  + [0, 1, 2, 3, 4]
        res['caption'] = res['caption'] + [PROMPT.format(caption = df.iloc[j]['s0'])]
        res['caption'] = res['caption'] + [PROMPT.format(caption = df.iloc[j]['s1'])]
        res['caption'] = res['caption'] + [PROMPT.format(caption = df.iloc[j]['s2'])]
        res['caption'] = res['caption'] + [PROMPT.format(caption = df.iloc[j]['s3'])]
        res['caption'] = res['caption'] + [PROMPT.format(caption = df.iloc[j]['s4'])]

    df = pd.DataFrame(res)
    df.to_csv(args.output_csv, index = False)

if __name__ == "__main__":
    main()