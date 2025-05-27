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

from verl import DataProto
from verl.utils.reward_score import (
    _default_compute_score,
    _extract_verifiable_part_of_solution,
)
import torch
from collections import defaultdict, Counter
import numpy as np
from typing import Tuple, List, Dict
from tqdm import tqdm


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def get_prompt_and_response_and_ground_truth(
        self,
        data_item,
    ) -> Tuple[str, str, str]:
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

        assert isinstance(ground_truth, str)

        return prompt_str, response_str, ground_truth
    
    def calculate_fraction_of_data_at_pass_rate_interval(
        self,
        all_pass_rates: List[float],
        pass_rate_interval: Tuple[int, int],
    ) -> float:
        num_pass_rates_between_interval = 0.0
        low = pass_rate_interval[0]
        high = pass_rate_interval[1]

        for pass_rate in all_pass_rates:
            if pass_rate >= low and pass_rate <= high:
                num_pass_rates_between_interval += 1.0

        return num_pass_rates_between_interval / len(all_pass_rates)
    
    def calculate_pass_rate_distribution(
        self,
        data: DataProto,
    ) -> Dict:
        prompt_to_correctness = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]
            (
                prompt_str, 
                response_str, 
                ground_truth,
            ) = self.get_prompt_and_response_and_ground_truth(
                data_item=data_item,
            )

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            prompt_to_correctness[prompt_str].append(score)

        prompt_to_pass_rate = {}
        for prompt in prompt_to_correctness:
            prompt_to_pass_rate[prompt] = np.mean(prompt_to_correctness[prompt])

        all_pass_rates = []
        for prompt in prompt_to_pass_rate:
            all_pass_rates.append(prompt_to_pass_rate[prompt])

        pass_rate_intervals = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        pass_rate_metrics = {}
        for pass_rate_interval in pass_rate_intervals:
            pass_rate_metrics[f"train_pass_rates/between_{pass_rate_interval}"] = (
                self.calculate_fraction_of_data_at_pass_rate_interval(
                    all_pass_rates=all_pass_rates,
                    pass_rate_interval=pass_rate_interval,
                )
            )

        return pass_rate_metrics

    def preprocess_plotting_matrices(self, prompt_list, data_source_list, gt_list, extracted_answer_list, llm_response_list):
        """
        Args:
            prompt_list: list of prompts
            data_source_list: list of data sources
            gt_list: list of ground truths
            extracted_answer_list: list of extracted answers
            llm_response_list: list of llm responses    
        Returns:
            num_of_prompts: list of number of prompts
            accuracy_list: list of accuracy scores
            mode_accuracy_list: list of mode accuracy scores


        This helper method is for creating the necessary lists for plotting the threshold accuracy curves.

        The returned lists can be plotted directly.

        NOTE: currently, we take the mode over the *extracted answers*, and then calculate the accuracy based on the *extracted* answer (and not the raw llm output).
        This can be changed, but it is the only thing implemented for now.
        """


        print(len(prompt_list))
        N = len(set(prompt_list))
        n = len(prompt_list) // N

        assert n * N == len(prompt_list)

        prediction_matrix = np.array(extracted_answer_list).reshape(N, n)
        ground_truths_matrix = np.array(gt_list).reshape(N, n)
        data_source_matrix = np.array(data_source_list).reshape(N, n)
        llm_response_matrix = np.array(llm_response_list).reshape(N, n) # original llm response. We don't use them for now

        # the actual plotting code: 
        thresholds = np.arange(0, 1.01, 0.05)

        accuracy_list = []
        num_of_prompts = []
        mode_accuracy_list = []
        for threshold in tqdm(thresholds, desc="computing threshold plots"):
            num_prompts_ = 0
            per_prompt_accuracy = []
            mode_accuracy = []
            for i in range(N): # for each prompt
                ith_pred_ = prediction_matrix[i]
                ith_pred = [x for x in ith_pred_ if x != ""]
                ith_pred = [x for x in ith_pred if x is not None]
                if len(ith_pred) == 0:
                    continue
                counter = Counter(ith_pred)
                mode, count = counter.most_common(1)[0]
                try:
                    mode = "\\boxed{"+mode+"}"
                except:
                    pass
                p = count / len(ith_pred)
                gt = ground_truths_matrix[i][0]
                data_source = data_source_matrix[i][0]
                

                if p >= threshold: # ONLY if the percentage is greater than the threshold
                    gt_ = [gt] * len(ith_pred) # make the ground truth list
                    assert len(gt_) == len(ith_pred)
                    # print(f"Prompt {0}: {ith_pred}\n\n(GT: {gt})")
                    # accuracy = sum([self.compute_score(ground_truth=gt_[j], model_output=ith_pred[j]) for j in range(len(ith_pred))]) / len(ith_pred) # compute accuracy for each prompt
                    ith_pred = ["\\boxed{" + x + "}" for x in ith_pred]
                    accuracy = sum([self.compute_score(data_source=data_source, solution_str=ith_pred[j], ground_truth=gt) for j in range(len(ith_pred))]) / len(ith_pred) # compute accuracy for each prompt
                    
                    per_prompt_accuracy.append(accuracy)
                    num_prompts_ += 1
                    mode_accuracy.append(self.compute_score(solution_str=mode, ground_truth=gt, data_source=data_source))
                
            accuracy_list.append(sum(per_prompt_accuracy) / len(per_prompt_accuracy) if len(per_prompt_accuracy) > 0 else 0.0)
            mode_accuracy_list.append(sum(mode_accuracy) / len(mode_accuracy) if len(mode_accuracy) > 0 else 0.0)
            num_of_prompts.append(num_prompts_)
            
        # normalize num_of_prompts to be between 0 and 1
        num_of_prompts = [x / N for x in num_of_prompts]
            
        assert len(num_of_prompts) == len(thresholds)
        assert len(mode_accuracy_list) == len(thresholds)
        assert len(accuracy_list) == len(thresholds)

        return num_of_prompts, accuracy_list, mode_accuracy_list


    def get_modal_accuracy_per_dataset(self, prompt_list, data_source_list, gt_list, extracted_answer_list) -> defaultdict: 
        """
        This is a modified version of the the preprocess_plotting_matrices function.
        Args:
            prompt_list: list of prompts
            data_source_list: list of data sources
            gt_list: list of ground truths
            extracted_answer_list: list of extracted answers
        Returns:
            modal_accuracy_dict : defaultdict of modal accuracy scores

        NOTE: currently, we take the mode over the *extracted answers*, and then calculate the accuracy based on the *extracted* answer (and not the raw llm output).
        This can be changed, but it is the only thing implemented for now.
        """


        print(len(prompt_list))
        N = len(set(prompt_list))
        n = len(prompt_list) // N

        assert n * N == len(prompt_list)

        prediction_matrix = np.array(extracted_answer_list).reshape(N, n)
        ground_truths_matrix = np.array(gt_list).reshape(N, n)
        data_source_matrix = np.array(data_source_list).reshape(N, n)

        mode_accuracy = defaultdict(list)

        for i in range(N): # for each prompt
            ith_pred_ = prediction_matrix[i]
            ith_pred = [x for x in ith_pred_ if x != ""]
            ith_pred = [x for x in ith_pred if x is not None]
            gt = ground_truths_matrix[i][0]
            data_source = data_source_matrix[i][0]
            
            if len(ith_pred) == 0:
                mode = None
            else:
                counter = Counter(ith_pred)
                mode, _ = counter.most_common(1)[0]
                try:
                    mode = "\\boxed{"+mode+"}"
                except:
                    mode = None
            # _list.append(self.compute_score(solution_str=mode, ground_truth=gt, data_source=data_source))
            mode_accuracy[data_source].append(self.compute_score(solution_str=mode, ground_truth=gt, data_source=data_source))

        return mode_accuracy



    def __call__(self, data: DataProto, return_dict=False, log_threshold_plot=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) # (batch_response, sequence len)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        gt_list, prompt_list, data_source_list, extracted_answer_list, llm_response_list = [], [], [], [], []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )


            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)


            # New code for threshold plotting
            # append the data to the list for threshold_plotting
            extracted_answer = _extract_verifiable_part_of_solution(
                data_source=data_source,
                solution_str=response_str,
            )
            gt_list.append(ground_truth)
            prompt_list.append(prompt_str)
            data_source_list.append(data_source)
            extracted_answer_list.append(extracted_answer)
            llm_response_list.append(response_str)
        
        

        if return_dict:
            
            if log_threshold_plot:
                # Only if log_threshold_plot is True, we do the additional computation for plotting curves, otherwise we skip
                plot_dict = defaultdict(list)
                num_of_prompts, accuracy_list, mode_accuracy_list = self.preprocess_plotting_matrices(
                                                                prompt_list=prompt_list,
                                                                data_source_list=data_source_list,
                                                                gt_list=gt_list,
                                                                extracted_answer_list=extracted_answer_list,
                                                                llm_response_list=llm_response_list,)
                plot_dict["num_of_prompts"] = num_of_prompts
                plot_dict["accuracy_list"] = accuracy_list
                plot_dict["mode_accuracy_list"] = mode_accuracy_list
                
            else:
                plot_dict = None

            # get the modal accuracy per dataset and pass rate distribution
            modal_accuracy_dict = self.get_modal_accuracy_per_dataset(
                prompt_list=prompt_list,
                data_source_list=data_source_list,
                gt_list=gt_list,
                extracted_answer_list=extracted_answer_list,
            )
            pass_rate_distribution = self.calculate_pass_rate_distribution(
                data=data,
            )
            raw_responses = defaultdict(list)
            raw_responses['gt_list'] = gt_list
            raw_responses['prompt_list'] = prompt_list
            raw_responses['data_source_list'] = data_source_list
            raw_responses['extracted_answer_list'] = extracted_answer_list

            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "plot_dict": plot_dict, 
                "pass_rate_distribution": pass_rate_distribution,
                "modal_accuracy_dict": modal_accuracy_dict,
                "raw_responses": raw_responses,
            }
        else:
            return reward_tensor
