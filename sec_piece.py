import os
import json
import tqdm
from io import StringIO
import sentencepiece as spm


def main():
    path = "./train/"
    with open('text.txt', 'w') as f:
        for filename in tqdm.tqdm(os.listdir(path)):
            for line in open(path + filename, "r+"):
                json_data = json.load(StringIO(line))
                f.write("\n".join(json_data["source_tokens"]))

    spm.SentencePieceTrainer.train(input="/home/frak/code_rep/great/text.txt",
                                   user_defined_symbols="<sep>",
                                   model_prefix="m",
                                   vocab_size=8000,
                                   shuffle_input_sentence=True,
                                   input_sentence_size=300000)


if __name__ == '__main__':
    main()
