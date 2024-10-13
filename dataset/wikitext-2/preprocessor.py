import os
import re
import random


def _read_wiki(source_path, target_path):
    
    with open(source_path, 'r') as f:
        lines = f.readlines()

    clean_lines = [re.sub(r'(\s*\.\s*){2,}', ' ', line) for line in lines]

    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in clean_lines if len(line.split(' . ')) >= 2]
    
    # print(len(paragraphs[2]))
    # print(paragraphs[2])

    combined_sentences = []

    for sentences in paragraphs:
        # split each sentence in the middle to two parts
        for sentence in sentences:
            
            words = sentence.split()
            mid = len(words) // 2  # calculate the middle index
            
            # 分成两部分，并用"\t"连接
            part1 = ' '.join(words[:mid])
            part2 = ' '.join(words[mid:])
            combined_sentences.append(part1 + '\t' + part2)

    # print(len(combined_sentences))
    # print(combined_sentences)

    random.shuffle(combined_sentences)

    # store
    with open(target_path, 'w') as f:
        for sentence in combined_sentences:
            if len(sentence) <= 32: continue
            f.write(sentence + '\n')


_read_wiki("test.csv",  "test.txt")
_read_wiki("train.csv", "train.txt")

def merge_files(train_file, test_file, output_file):
    with open(output_file, 'w') as outfile:
        # 读取并写入train.txt内容
        with open(train_file, 'r') as infile:
            outfile.write(infile.read())
        
        # 读取并写入test.txt内容
        with open(test_file, 'r') as infile:
            outfile.write(infile.read())

# 合并train.txt和test.txt，生成data.txt
merge_files("train.txt", "test.txt", "data.txt")