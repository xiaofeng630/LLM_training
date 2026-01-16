import urllib.request 
import zipfile 
import os 
import json
from pathlib import Path
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" 
zip_path = "sms_spam_collection.zip" 
extracted_path = "sms_spam_collection" 
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

## 下载官方垃圾邮件分类数据集
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):     
    if data_file_path.exists():         
        print(f"{data_file_path} already exists. Skipping download and extraction.")         
        return
    
    with urllib.request.urlopen(url) as response:         
        with open(zip_path, "wb") as out_file:             
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref: 
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection" 
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

## 下载官方指令微调数据集
def download_and_load_file(file_path, url):     
    if not os.path.exists(file_path):         
        with urllib.request.urlopen(url) as response:             
            text_data = response.read().decode("utf-8")        
        with open(file_path, "w", encoding="utf-8") as file:             
            file.write(text_data)    
    else:          
        with open(file_path, "r", encoding="utf-8") as file: 
            text_data = file.read()     
    with open(file_path, "r") as file:
        data = json.load(file)     
    return data


if __name__ == "__main__":
    # download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    file_path = "instruction-data.json" 
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json")
    data = download_and_load_file(file_path, url) 
    print("Number of entries:", len(data))




