import pandas as pd 

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]      
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)     
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])     
    return balanced_df

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)     
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]     
    validation_df = df[train_end:validation_end]     
    test_df = df[validation_end:]
    return train_df, validation_df, test_df