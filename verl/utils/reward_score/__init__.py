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
# from . import gsm8k, math, prime_math, prime_code

from typing import Optional


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)

    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        # from . import math
        # res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)

    #
    # Commenting this chunk out since we use the MATH verifier/extractor for DAPO for now
    #
    
    # elif data_source == 'math_dapo' or data_source.startswith("aime"):
    #     from . import math_dapo
    #     res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)

    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)

    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)

    elif data_source == 'countdown':
        from . import countdown
        res = countdown.compute_score(solution_str, ground_truth)

    # For AIME, we use the same parser/solution extractor as MATH
    elif "AIME" in data_source:
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)

    # For DAPO, we use the same parser/solution extractor as MATH
    elif data_source == "math_dapo":
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)

    # for the six evaluation sources, MATH parser works just as well:
    elif data_source in ["math500", "aime24", "aime25", "minerva_math", "olympiadbench", "amc23", "aime"]: 
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)

    # For NUMINA-MATH, we also use the MATH parser:
    elif data_source == "numina_math":
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    

def _extract_verifiable_part_of_solution(
    data_source: str,
    solution_str: str,
) -> Optional[str]:
    if data_source == 'countdown':
        from . import countdown
        result = countdown.extract_solution(solution_str=solution_str)

    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        result = gsm8k.extract_solution(solution_str=solution_str)

    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math_verify
        result = math_verify.extract_solution(solution_str=solution_str)

    # For AIME, we use the same parser/solution extractor as MATH
    elif "AIME" in data_source:
        from . import math_verify
        result = math_verify.extract_solution(solution_str=solution_str)

    # For DAPO, we use the same parser/solution extractor as MATH
    elif data_source == "math_dapo":
        from . import math_verify
        result = math_verify.extract_solution(solution_str=solution_str)

    # for the six evaluation sources, MATH parser works just as well:
    elif data_source in ["math500", "aime24", "aime25", "minerva_math", "olympiadbench", "amc23"]: 
        from . import math_verify
        result = math_verify.extract_solution(solution_str=solution_str)

    # For NUMINA-MATH, we also use the MATH parser:
    elif data_source == "numina_math":
        from . import math_verify
        result = math_verify.extract_solution(solution_str=solution_str)

    else:
        raise NotImplementedError(f"Solution extraction method not implemented for data source {data_source}")
    
    return result
