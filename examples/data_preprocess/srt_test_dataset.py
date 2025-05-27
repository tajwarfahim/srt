# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the compiled test dataset from the following paper:

         "Can Large Reasoning Models Self-Train?"

Project website: https://self-rewarding-llm-training.github.io/
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument(
        "--data_source", 
        default="ftajwar/srt_test_dataset", 
        help="The dataset to be used. Default one contains all six of them",
    )

    args = parser.parse_args()

    data_source = args.data_source
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('Problem')

            question = question + ' ' + instruction_following

            solution = example.pop('Answer')
            solution = str(solution)
            data_source = example.pop('data_source')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    hdfs_dir = args.hdfs_dir
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=args.local_dir, dst=hdfs_dir)
