import os
import json
import tqdm
from io import StringIO
from collections import defaultdict


def main():
    path = "./train/"
    token_dict = defaultdict(int)
    for filename in tqdm.tqdm(os.listdir(path)):
        for line in open(path + filename, "r+"):
            json_data = json.load(StringIO(line))
            for tokens in json_data["source_tokens"]:
                tokens = "".join(filter(lambda x: x not in "\n\r`\'", tokens))
                tokens = tokens.replace("_", " ")
                for token in tokens.split():
                    token_dict[token] += 1

    set_tokens = set()
    for token, c in token_dict.items():
        if c > 19:
            set_tokens.add(token)

    print(len(set_tokens))
    f = open("single_tokens_vocab.txt", "a")
    f.write("\n".join(set_tokens))
    f.close()


if __name__ == '__main__':
    main()
