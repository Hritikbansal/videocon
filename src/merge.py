import pandas as pd

## merge train
file1 = 'data/train_llm_entailment.csv'
file2 = 'data/train_llm_feedback.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
print(df1.columns, df2.columns)
df = pd.concat([df1, df2])
df = df.sample(frac = 1)
print(len(df))

df.to_csv('data/train_llm_mix_entail_feedback.csv', index = False)

## merge val
file1 = 'data/val_llm_entailment.csv'
file2 = 'data/val_llm_feedback.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
print(df1.columns, df2.columns)
df = pd.concat([df1, df2])
df = df.sample(frac = 1)
print(len(df))

df.to_csv('data/val_llm_mix_entail_feedback.csv', index = False)


## merge test
file1 = 'data/test_llm_entailment.csv'
file2 = 'data/test_llm_feedback.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
print(df1.columns, df2.columns)
df = pd.concat([df1, df2])
df = df.sample(frac = 1)
print(len(df))

df.to_csv('data/test_llm_mix_entail_feedback.csv', index = False)