from dataset import load_dataset, DatasetConfig, construct_prompt
import numpy as np
from copy import deepcopy
from scipy.special import entr
import argparse
from vllm import LLM, SamplingParams

GREEDY_PARAMS = {
    "temperature": 0.0, "top_p": 1.0, "max_tokens": 100, "n": 1,
    "max_tokens":1, 
}

### 这一部分来自于g_fair_prompting

def get_p_content_free(dataset_config:DatasetConfig, train_sentences, train_labels, model: LLM, content_free_inputs=('N/A',)):
    """Query model with content free input, return its prediction probability for each label"""
    label_dict = dataset_config.label_dict

    all_p_y = []
    prompts = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt(dataset_config, train_sentences, train_labels, content_free_input)
        for i, answers in label_dict.items():
            for a in answers:
                prompts.append(prompt + " " + a)
    sampling_params = SamplingParams(**GREEDY_PARAMS)
    sampling_params.prompt_logprobs = 0
    outputs = model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    last_token_probs = [
        np.exp(list(o.prompt_logprobs[-1].values())[0]) for o in outputs
    ]
    prob_iter = iter(last_token_probs)
    for content_free_input in content_free_inputs:
        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += next(prob_iter)
            p_y[i] = prob
        all_p_y.append(p_y)
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y_norm = p_y / np.sum(p_y)
    return p_y_norm

def find_fair_demonstration(all_train_sentences, all_train_labels, num_shots, dataset_config: DatasetConfig, model: LLM):
    def loop_list(list_origin, saved_idx, current_idx):
        list_tmp = [deepcopy(list_origin[i]) for i in(saved_idx)]
        list_tmp.insert(0, deepcopy(list_origin[current_idx]))
        return list_tmp
    
    def cal_fair(p_cf):
        pcf_norm = np.array(p_cf).T/np.sum(np.array(p_cf), axis=1)
        pcf_entropy = entr(pcf_norm.T).sum(axis=1)
        max_idx = np.argmax(pcf_entropy)
        max_value = np.max(pcf_entropy)
        return max_idx, max_value
    
    train_sentences, train_labels= [], []
    for i in range(int(num_shots/4)):
        idxs = np.random.choice(len(all_train_labels), size=4, replace=False)
        train_sentences += [deepcopy(all_train_sentences[i]) for i in idxs]
        train_labels += [deepcopy(all_train_labels[i]) for i in idxs]

    content_free_inputs = ["N/A", "", "[MASK]"]
    all_loop = len(train_labels)
    fair_idx, fair_value = [], 0
    for loop in range(all_loop):
        all_label_probs_loops, p_cf_loops, p_cf_loops_nonorm, remain_idx = [], [], [], []
        for loop_app in range(all_loop):
            train_sentences_tmp, train_labels_tmp = deepcopy(train_sentences), deepcopy(train_labels)
            if loop_app in fair_idx:
                continue
            remain_idx.append(loop_app)
            train_sentences_tmp = loop_list(train_sentences_tmp, fair_idx, loop_app)
            train_labels_tmp = loop_list(train_labels_tmp, fair_idx, loop_app)
            p_cf_loops.append(
                # get_p_content_free(params, train_sentences_tmp, train_labels_tmp, content_free_inputs=content_free_inputs, tokenizer=tokenizer, model_bloom=model_bloom)
                # 用vllm重写了一个
                get_p_content_free(dataset_config, train_sentences_tmp, train_labels_tmp, model, content_free_inputs)
            )
        max_fair_idx, max_fair_value = cal_fair(p_cf_loops)
        if max_fair_value < fair_value:
            break
        fair_idx.insert(0,remain_idx[max_fair_idx])
        fair_value = max_fair_value
    train_sentences_tmp = [train_sentences[i] for i in fair_idx]
    train_labels_tmp = [train_labels[i] for i in fair_idx]
    return train_sentences_tmp, train_labels_tmp


### 下面是我自己写的

def perpare_prompts(dataset, model, args):
    train_sentences_tmp, train_labels_tmp = find_fair_demonstration(
        all_train_sentences=dataset["train_sentences"],
        all_train_labels=dataset["train_labels"],
        num_shots=args.num_shots,
        dataset_config=dataset["config"],
        model=model
    )
    prompts = [
        construct_prompt(
            dataset_config=dataset["config"],
            train_sentences=train_sentences_tmp,
            train_labels=train_labels_tmp,
            test_sentence=t
        )
        for t in dataset["test_sentences"]
    ]
    return prompts

def get_target_token_ids(dataset_config: DatasetConfig, model: LLM):
    label_dict = dataset_config.label_dict
    tokenizer = model.get_tokenizer()
    target_token_ids = [
        [tokenizer.encode(a, add_special_tokens=False)[0] for a in answers]
        for answers in label_dict.values()
    ]
    return target_token_ids

def evaluate(responses, test_labels, target_token_ids):
    all_label_probs = []
    for re in responses:
        logprob = re.outputs[0].logprobs[0]
        label_probs = []
        for token_ids in target_token_ids:
            log_prob = [logprob.get(t_id, -float("inf")) for t_id in token_ids]
            label_probs.append(np.sum(np.exp(log_prob)))
        prob = np.array(label_probs) / np.sum(label_probs)
        all_label_probs.append(prob)

    prediction = np.argmax(all_label_probs, axis=1)
    ground_truth = np.array(test_labels)
    accuracy = np.mean(prediction == ground_truth)
    return accuracy

def main(args):

    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85, max_model_len=2048 # avoid OOM
    )
    for dataset in args.dataset:
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels, dataset_config = load_dataset(
            dataset = dataset,
        )

        if args.do_debug:
            all_test_sentences = all_test_sentences[:100]
            all_test_labels = all_test_labels[:100]
            print("Debug mode, only use first 100 samples of the dataset")

        dataset = {
            "config": dataset_config,
            "train_sentences": all_train_sentences,
            "train_labels": all_train_labels,
            "test_sentences": all_test_sentences,
            "test_labels": all_test_labels
        }

        prompts = perpare_prompts(dataset, model, args)

        print(f"PROMPT EXAMPLE: \n{prompts[0]}\n")

        sampling_params = SamplingParams(**GREEDY_PARAMS)
        sampling_params.logprobs=5000
        responses = model.generate(prompts=prompts, sampling_params=sampling_params)

        target_token_ids = get_target_token_ids(dataset_config, model)
        accuracy = evaluate(responses, dataset["test_labels"], target_token_ids)
        print(f"Dataset: {dataset}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model_name_or_path", type=str, default="gpt2-medium")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--dataset", nargs="+", type=str, required=True,
                        choices=["sst2", "agnews", "trec", "rte", "cola"])
    parser.add_argument("--do_debug", action="store_true", help="If set, only use first 100 samples of the dataset")
    args = parser.parse_args()
    main(args)
    # print(f"Dataset: {args.dataset}, Accuracy: {results:.4f}")

