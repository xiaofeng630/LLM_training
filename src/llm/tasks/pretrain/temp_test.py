import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")
print(tokenizer.max_token_value)