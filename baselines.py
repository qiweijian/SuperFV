from vllm import LLM, SamplingParams

from fair_prompt import load_dataset, get_target_token_ids, GREEDY_PARAMS, evaluate, construct_prompt

from typing import Literal

import csv

def load_dataset_wrap(dataset_name: Literal["sst2", "agnews", "trec", "rte", "cola"]):
    all_train_sentences, all_train_labels, all_test_sentences, all_test_labels, dataset_config = load_dataset(
        dataset=dataset_name,
    )
    return {
        "config": dataset_config,
        "train_sentences": all_train_sentences,
        "train_labels": all_train_labels,
        "test_sentences": all_test_sentences,
        "test_labels": all_test_labels
    }

def generate_and_evaluate(llm, prompts, labels, dataset_config):
    sampling_params = SamplingParams(**GREEDY_PARAMS)
    sampling_params.logprobs=5000
    responses = llm.generate(prompts=prompts, sampling_params=sampling_params)

    target_token_ids = get_target_token_ids(dataset_config, llm)
    return evaluate(
        responses=responses,
        test_labels=labels,
        target_token_ids=target_token_ids,
    )

import numpy as np
from copy import deepcopy

def create_n_shot_prompts(dataset, n_shot, seed):
    prompts = []
    np.random.seed(seed)
    for t in dataset["test_sentences"]:
        idxs = np.random.choice(len(dataset['train_labels']), size=n_shot, replace=False)
        train_sentences = [deepcopy(dataset['train_sentences'][i]) for i in idxs]
        train_labels = [deepcopy(dataset['train_labels'][i]) for i in idxs]
        prompt = construct_prompt(
            dataset_config=dataset['config'],
            train_sentences=train_sentences,
            train_labels=train_labels,
            test_sentence=t,
        )
        prompts.append(prompt)
    return prompts


if __name__=="__main__":
    llm = LLM(
        model = "/data/MODELS/llama-2-7b",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85, max_model_len=2048
    )

    SEED = 42

    with open("baselines_results2.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "n_shot", "accuracy"])
        for dataset_name in ["trec", "rte", "cola"]:
            dataset = load_dataset_wrap(dataset_name)
            print(f"{dataset_name}:\tTRAIN: {len(dataset['train_sentences'])}, TEST: {len(dataset['test_sentences'])}")
            for n_shot in [0, 1, 5, 10]:
                print(f"n_shot={n_shot}")
                prompts = create_n_shot_prompts(dataset, n_shot, seed=SEED)
                acc = generate_and_evaluate(llm, prompts, dataset['test_labels'], dataset['config'])
                print(f"Accuracy: {acc}")
                writer.writerow([dataset_name, n_shot, acc])