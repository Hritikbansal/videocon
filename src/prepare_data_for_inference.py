import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--input_csv', type = str, required = True, help = 'input csv file')
parser.add_argument('--output_csv', type = str, required = True, help = 'output csv file')
parser.add_argument('--task', type = str, default = 'entailment', help = 'task')

args = parser.parse_args()

PROMPT = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Does this video entail the description: "{caption}"?
AI: '''

if args.task == 'entailment':
    df = pd.read_csv(args.input_csv) ## should contain videopath and text as a field
    captions = []
    for j in range(len(df)):
        captions.append(PROMPT.format(caption = df.iloc[j]['text']))
    df['caption'] = captions
    df.to_csv(args.output_csv, index = False)