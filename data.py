import pandas as pd

# !python -m wget 'https://storage.googleapis.com/samanantar-public/V0.2/data/en2indic/en-mr.zip'
# !python -m wget 'https://storage.googleapis.com/samanantar-public/V0.2/data/en2indic/en-hi.zip'

# !unzip en-hi.zip

# f = open(trainfileEH, "r")
# EnglishV = f.readlines()
# f.close()
# EnglishV = removeNewLineChars(EnglishV)
# print(f"Total English Sentences : {len(EnglishV)}")

# f = open(trainfileEH, "r")
# HindiV = f.readlines()
# f.close()
# HindiV = removeNewLineChars(HindiV)
# print(f"Total Hindi Sentences : {len(EnglishV)}")

# train_df = pd.DataFrame(list(zip(EnglishV, HindiV)), columns=["english_sentence", 'hindi_sentence'])


def getData():
    train_df = pd.read_csv(
        "https://raw.githubusercontent.com/saurabhburewar/ML_English-to-Hindi-Language-translation/main/data/Hindi_English_Truncated_Corpus.csv")
    train_df.drop(['source'], axis=1, inplace=True)
    for index, row in train_df.iterrows():
        if len(str(row['english_sentence'])) < 20 and len(str(row['english_sentence'])) > 200:
            train_df.drop(index=index, inplace=True)
        else:
            row['english_sentence'] = "<SOS> " + \
                str(row['english_sentence']) + " <EOS>"
            row['hindi_sentence'] = "<SOS> " + row['hindi_sentence'] + " <EOS>"
    train_df = train_df.sample(64000, random_state=1)
    train_df.head()

    return train_df
