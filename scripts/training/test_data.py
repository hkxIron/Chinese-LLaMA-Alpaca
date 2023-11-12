import os
from itertools import chain
from pathlib import Path
from typing import List

import pyarrow
import transformers
from datasets import load_dataset, concatenate_datasets
from datasets.formatting.formatting import LazyBatch
from transformers import (
    LlamaTokenizer,
)
from transformers.testing_utils import CaptureLogger


def test_pretrain_data():
    tokenizer_kwargs = {
        "cache_dir": "temp_data_cache_dir",
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": False
    }
    #tokenizer_name='chinese-llama-lora-7b'
    tokenizer_name='../../tokenizer/chs-llama'
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # examples:{'text': Value(dtype='string', id=None)}
    block_size = 1024
    def tokenize_function(examples:LazyBatch):
        # 注意：要进入此函数，需要将dataset的cache文件删除
        with CaptureLogger(tok_logger) as cl:
            # examples是pyarrow.Table, 只有一列，列名为"text"
            # examples['text']为list[str]，其中list长度为1000
            data:List[str] = examples['text']
            output = tokenizer(data)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output


    # examples: {'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None), 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None)}
    def group_texts(examples:LazyBatch):
        # 注意：要进入此函数，需要将dataset的cache文件删除
        # Concatenate all texts, 将所有1000个样本的input_ids连在一起. input_ids:List[str], attention_mask:List[int]
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]]) # list(examples.keys())[0] = 'input_ids'
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len. 将总长度按block=1024进行切分,即不相关的样本也会混在一起进行训练，而不是像以前一样用padding处理
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # labels 与input_ids的值相同
        result["labels"] = result["input_ids"].copy()
        return result

    dataset_dir='../../data/'
    lm_datasets = []
    path = Path(dataset_dir)
    files = [file.name for file in path.glob("*.txt")]
    #files = ['../../data/pt_sample_data.txt']
    data_cache_dir = 'temp_data_cache_dir'
    for idx, file in enumerate(files):
        data_file = os.path.join(path, file)
        filename = ''.join(file.split(".")[:-1])

        #processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
        cache_dir = os.path.join(data_cache_dir, filename + "_text")
        os.makedirs(cache_dir, exist_ok=True)
        raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
        print(f"{file} has been loaded")
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns="text",
            load_from_cache_file=True,
            keep_in_memory=False,
            cache_file_names={k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
            desc="Running tokenizer on dataset",
        )
        grouped_datasets = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=1,
            load_from_cache_file=True,
            keep_in_memory=False,
            cache_file_names={k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
            desc=f"Grouping texts in chunks of {block_size}",
        )
        processed_dataset = grouped_datasets
        processed_dataset.save_to_disk(cache_dir)

        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

    lm_datasets = lm_datasets.train_test_split(test_size = 0.1)

    train_dataset = lm_datasets['train']
    max_train_samples = 1000
    if max_train_samples is not None:
        max_train_samples = min(len(train_dataset), max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    print(f"Num train_samples:{len(train_dataset)}")
    print("training example:")
    for index in range(5):
        sample=train_dataset[index]
        print("dataset keys:", sample.keys()) # dict_keys(['input_ids', 'attention_mask', 'labels'])

        print("tokenizer attention_mask:", sample['attention_mask'], " attention mask size:", len(sample['attention_mask'])) # mask size:1024
        print("tokenizer input ids:",sample['input_ids'], " len:", len(sample['input_ids'])) # input_ids len:1024

        print("tokenizer input text:[",tokenizer.decode(sample['input_ids']), "] text len:", len(tokenizer.decode(sample['input_ids']))) # 1709
        print("tokenizer label text:[",tokenizer.decode(sample['labels']), "] label text len:", len(tokenizer.decode(sample['labels']))) # 1709
        # input_ids与labels内容一样，但不是同一个对象
        assert sample['input_ids'] == sample['labels']
        assert id(sample['input_ids']) != id(sample['labels'])
        print("="*10+"\n\n")

if __name__ == "__main__":
    test_pretrain_data()