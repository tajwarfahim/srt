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
Preprocess DAPO dataset to parquet format

This file can be used to create an unlabeled version of the same dataset,
to be used for self-rewarding training (SRT).
"""

import os
import datasets
import numpy as np

from verl.utils.hdfs_io import copy, makedirs
import argparse


GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED = "LABEL_BY_SELF_CONSISTENCY"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--label_noise", type=float, default=0.1)
    parser.add_argument("--add_self_consistency_labels", action='store_true')
    parser.add_argument('--dataset_path', type=str, default='ftajwar/deduplicated_dapo_dataset')

    args = parser.parse_args()

    data_source = 'math_dapo'

    dataset_path = args.dataset_path
    dataset = datasets.load_dataset(dataset_path, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn_train(split, label_noise):

        def process_fn(example, idx):
            question = example.pop('prompt')

            question = question + ' ' + instruction_following

            original_solution = example.pop('answer')
            original_solution = str(original_solution)

            # add noise
            random_num = np.random.uniform(low=0.0, high=1.0, size=None)

            if random_num <= label_noise:
                print("Original solution: ", original_solution)

                if args.add_self_consistency_labels:
                    solution = GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED
                else: 
                    solution = str(int(original_solution) + 1)
                    
                print("Label noise solution: ", solution)

            else:
                solution = original_solution

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                    "solution_hidden_during_training": original_solution,
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return lambda example, idx: process_fn(example=example, idx=idx)
    
    def make_map_fn_test(split):

        def process_fn(example, idx):
            question = example.pop('prompt')

            question = question + ' ' + instruction_following

            solution = example.pop('answer')
            solution = str(solution)

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                    "solution_hidden_during_training": solution,
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn_train('train', label_noise=args.label_noise), 
        with_indices=True,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn_test('test'), 
        with_indices=True,
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
