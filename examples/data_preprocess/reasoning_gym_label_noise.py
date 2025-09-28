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
import os
import json
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED = "LABEL_BY_SELF_CONSISTENCY"
def json_safe(obj):
    """
    Return a JSON string that faithfully represents `obj`, but never
    raises TypeError (e.g. when keys are not str).  We stringify any
    non-str key with a prefix so collisions are impossible.
    """
    if isinstance(obj, dict):
        fixed = {}
        for k, v in obj.items():
            if isinstance(k, str):
                nk = k
            else:
                nk = f"@{repr(k)}"        #  ➜  "@(2, 3)" or "@0"
            # if the original dict *already* had that string key,
            # append a numeric suffix to stay unique
            i = 1
            while nk in fixed:
                nk = f"{nk}_{i}"
                i += 1
            fixed[nk] = json_safe(v)
        return fixed
    elif isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    else:
        return obj
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument(
        "--data_source", 
        default="reasoning_gym",  
        help="The dataset to be used. Default one contains all six of them",
    )
    parser.add_argument("" \
        "--repo_name",
        )

    args = parser.parse_args()

    data_source = args.data_source
    print(f"Loading the {args.repo_name} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(args.repo_name, trust_remote_code=True)

    train_dataset = dataset['train']
    # assert "test" in dataset, "The dataset must contain a 'test' split."
    # test_dataset = dataset['test']


    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('question')

            question = question + ' ' + instruction_following

            
            metadata = example.pop('metadata')
            # breakpoint()
            _metadata = json.loads(metadata)
            _metadata["answer"] = GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED
            _m = json.dumps(json_safe(_metadata), ensure_ascii=False)

            data = {
                "data_source": args.data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "puzzles",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": _m,
                    "solution_hidden_during_training": metadata, # metadata contains the ground truth
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    
    if not args.local_dir: 
        local_dir = "dataset/reasoning_gym/" + args.repo_name.split('/')[-1] + "_noised"
    else: 
        local_dir = args.local_dir + args.repo_name.split('/')[-1] + "_noised"
    hdfs_dir = args.hdfs_dir
    
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))


    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=args.local_dir, dst=hdfs_dir)
