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

## 用于快速查看数据文件结构
def view_parquet(parquet_file):
    df = pd.read_parquet(parquet_file)
    counts = df["label"].value_counts()
    print(counts)
    
    # i = 0
    # for i in range(10):
    #     print(df['text'][i])
    #     print(type(df['label'][i]))

    # print(df.columns)
    # print(df["text"][0])
    # print(df["label"][0])

def parquet_overview(parquet_file):
    """
    快速查看 parquet 的整体结构
    """
    df = pd.read_parquet(parquet_file)

    print("========== Basic Info ==========")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("")

    print("========== Columns & Dtypes ==========")
    print(df.dtypes)

def preview_rows(parquet_file, n=5):
    """
    查看前 n 条样本
    """
    df = pd.read_parquet(parquet_file)
    print(df.head(n))

def label_distribution(parquet_file, label_col="label"):
    """
    查看标签分布
    """
    df = pd.read_parquet(parquet_file)

    if label_col not in df.columns:
        print(f"[ERROR] Column '{label_col}' not found.")
        return

    print("========== Label Distribution ==========")
    print(df[label_col].value_counts())

def label_ratio(parquet_file, label_col="label"):
    """
    查看标签比例
    """
    df = pd.read_parquet(parquet_file)
    ratios = df[label_col].value_counts(normalize=True)

    print("========== Label Ratio ==========")
    print((ratios * 100).round(2).astype(str) + "%")

def text_length_stats(parquet_file, text_col="text"):
    """
    统计文本长度（字符级）
    """
    df = pd.read_parquet(parquet_file)

    lengths = df[text_col].astype(str).apply(len)

    print("========== Text Length Stats ==========")
    print(f"Min: {lengths.min()}")
    print(f"Max: {lengths.max()}")
    print(f"Mean: {lengths.mean():.2f}")
    print(f"Median: {lengths.median()}")

def extreme_text_samples(parquet_file, text_col="text"):
    """
    查看最长和最短的文本
    """
    df = pd.read_parquet(parquet_file)
    lengths = df[text_col].astype(str).apply(len)

    shortest = df.loc[lengths.idxmin()]
    longest = df.loc[lengths.idxmax()]

    print("========== Shortest Text ==========")
    print(shortest[text_col])
    print("\n========== Longest Text ==========")
    print(longest[text_col])


if __name__ == "__main__":
    view_parquet("/home/hjzd/lzz/LLM_training/data/classification/ChnSentiCorp/test.parquet")