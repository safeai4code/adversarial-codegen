import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from evalplus.codegen import run_codegen
from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import (
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec
from termcolor import cprint
from tqdm import tqdm

Result = Tuple[str, List[bool]]


def get_groundtruth(problems, hashcode, tasks_only_output_not_none):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        print(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Computing expected output...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )

        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output


def add_identifier(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for i, sample in enumerate(samples):
        sample["_identifier"] = (
            sample["task_id"] + f"_{i}"
        )
    return samples


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    base_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        dataset,
        solution,
        problem["base_input"],
        problem["entry_point"],
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )

    if not base_only:
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

    return ret


def evaluator(
    dataset: str,
    samples: List[dict] = None,
    base_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default"
):
    
    assert samples is not None, "No samples provided"

    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    # TODO: We can further improve the logic here: Move to attack_framework.py
    # if result_file is None:
    #     print("Warning: No result file specified. Results are only exihibted.")
    # else:
        # get the result_path from the result_file
        # assert output_dir is not None, "You should provide output_dir"
        # result_path = os.path.join(output_dir, result_file)


    if dataset == "humaneval":
        problems = get_human_eval_plus(
            mini=mini, noextreme=noextreme, version=version
        )
        dataset_hash = get_human_eval_plus_hash(
            mini=mini, noextreme=noextreme, version=version
        )
        expected_output = get_groundtruth(problems, dataset_hash, [])
    elif dataset == "mbpp":
        problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)
        dataset_hash = get_mbpp_plus_hash(
            mini=mini, noextreme=noextreme, version=version
        )
        expected_output = get_groundtruth(
            problems,
            dataset_hash,
            MBPP_OUTPUT_NOT_NONE_TASKS,
        )

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hash": dataset_hash,
        "eval": {},
    }

    samples = add_identifier(samples)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        eval_results = defaultdict(list)  # task_id ->
        remainings = set()

        print("Reading samples...")
        for sample in tqdm(samples):
            task_id = sample["task_id"]
            if task_id not in problems:
                warn(
                    f"Task {task_id} is found in the samples but not found in the dataset"
                )
                continue
            solution = (
                sample["solution"]
                if "solution" in sample
                else problems[task_id]["prompt"] + sample["completion"]
            )
            remainings.add(sample["_identifier"])
            args = (
                dataset,
                completion_id[task_id],
                problems[task_id],
                solution,
                expected_output[task_id],
                base_only,
                not test_details,  # fast_check
                sample["_identifier"],
                min_time_limit,
                gt_time_limit_factor,
            )
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        assert n_samples == len(remainings), "Missing problems in unfinished"
        # assert len(completion_id) == len(problems), "Missing problems in samples"

        def stucking_checker():
            while remainings:
                last_size = len(remainings)
                time.sleep(20)
                if last_size != len(remainings) or len(remainings) == 0:
                    continue
                # Potential stucking
                warn("No samples had finished testing in the last 20s")
                warn(f"{len(remainings)} samples to be tested: {remainings}")

        threading.Thread(target=stucking_checker).start()

        for future in tqdm(as_completed(futures), total=n_samples):
            result = future.result()
            remainings.remove(result["_identifier"])
            eval_results[result["task_id"]].append(result)

    # sort the results for each problem by completion_id
    for task_id, task_results in eval_results.items():
        task_results.sort(key=lambda x: x["completion_id"])
        results["eval"][task_id] = []
        for res in task_results:

            def get_failed_tests(stat, details, inputs) -> List[Any]:
                if stat == PASS or not details:
                    return []

                if test_details:
                    return [
                        inputs[i] for i in range(len(details)) if not details[i]
                    ]

                # else => simply return the only and the last fail test
                return [inputs[len(details) - 1]]

            base_stat, base_details = res["base"]
            base_fail_tests = get_failed_tests(
                base_stat, base_details, problems[task_id]["base_input"]
            )

            # initialize plus tests
            plus_stat = None
            plus_fail_tests = []

            # with plus tests
            if not base_only:
                plus_stat, plus_details = res["plus"]
                plus_fail_tests = get_failed_tests(
                    plus_stat, plus_details, problems[task_id]["plus_input"]
                )

            if dataset == "mbpp":
                base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

            results["eval"][task_id].append(
                {
                    "task_id": task_id,
                    "solution": res["solution"],
                    "base_status": base_stat,
                    "plus_status": plus_stat,
                    "base_fail_tests": base_fail_tests,
                    "plus_fail_tests": plus_fail_tests,
                }
            )

    # Calculate pass@k.
    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not base_only:
            new_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )
    base_correct = np.array(base_correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1, 10, 100]
        if total.min() >= k
    }
    cprint(f"{dataset} (base tests)", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")
    results["pass_at_k"] = {"base": pass_at_k}

    if new_correct:
        cprint(f"{dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1, 10, 100]
            if (total >= k).all()
        }
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "green")
        results["pass_at_k"]["plus"] = pass_at_k

    return results


# Example usage
if __name__ == "__main__":
    
    dataset = 'mbpp'
    
    # Load appropriate dataset
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
    else:  # mbpp
        from evalplus.data import get_mbpp_plus
        problems = get_mbpp_plus()
    
    # Load generations
    generations = [
        {
            "task_id": "Mbpp/2",
            "solution": "def similar_elements(list1, list2):\n    return set(list1).intersection(set(list2))\n",
        },
    ]
    test_file = "/home/sfang9/workshop/aisec/adversarial-attack-nlp/original_prompts.jsonl"
    with open(test_file, "r") as f:
        generations = [json.loads(line) for line in f]
    
    
    # Run evaluation
    results = evaluator(
        dataset=dataset,
        samples=generations,
    )

    print(len(results["eval"]))
    print(results["eval"]["Mbpp/2"])
