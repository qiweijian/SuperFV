import pandas as pd
import json
import numpy as np
from copy import deepcopy
from dataclasses import dataclass

ROOT_DIR = './data'

def load_cola():
    train_sentences, train_label, test_sentences, test_label = [], [], [], []
    file2 = open(f"{ROOT_DIR}/glue/CoLA/train.txt", "r", encoding="utf-8")
    for line in file2.readlines():
        line = line.strip('\n').split('\t')
        train_sentences.append(line[0])
        train_label.append(int(line[1]))
    filet = open(f"{ROOT_DIR}/glue/CoLA/dev.txt", "r", encoding="utf-8")
    for linet in filet.readlines():
        linet = linet.strip('\n').split('\t')
        test_sentences.append(linet[0])
        test_label.append(int(linet[1]))
    return train_sentences, train_label, test_sentences, test_label

def load_sst2():
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.train", "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.test", "r", encoding="utf-8") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)
    return train_sentences, train_labels, test_sentences, test_labels

def load_agnews():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/train.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_trec():
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'{ROOT_DIR}/data/trec/train.txt', 'r', encoding="utf-8") as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'{ROOT_DIR}/data/trec/test.txt', 'r', encoding="utf-8") as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    return train_sentences, train_labels, test_sentences, test_labels

def load_rte():
    train_questions = []
    train_answers = []
    with open(f"{ROOT_DIR}/data/rte/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    test_questions = []
    test_answers = []
    with open(f"{ROOT_DIR}/data/rte/val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    return train_questions, train_answers, test_questions, test_answers

@dataclass
class DatasetConfig:
    dataset: str
    prompt_prefix: str
    q_prefix: str
    a_prefix: str
    label_dict: dict
    inv_label_dict: dict
    task_format: str
    num_tokens_to_predict: int

def load_dataset(dataset):

    # if params['dataset'] == 'sst2':
    if dataset == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        # params['prompt_prefix'] = ""
        # params["q_prefix"] = "Review: "
        # params["a_prefix"] = "Sentiment: "
        # params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        # params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        # params['task_format'] = 'classification'
        # params['num_tokens_to_predict'] = 1
        dataset_config = DatasetConfig(
            dataset='sst2',
            prompt_prefix="",
            q_prefix="Review: ",
            a_prefix="Sentiment: ",
            label_dict={0: ['Negative'], 1: ['Positive']},
            inv_label_dict={'Negative': 0, 'Positive': 1},
            task_format='classification',
            num_tokens_to_predict=1
        )
    elif dataset == 'agnews':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        dataset_config = DatasetConfig(
            dataset='agnews',
            prompt_prefix="Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n",
            q_prefix="Article: ",
            a_prefix="Answer: ",
            label_dict={0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology', 'Science']},
            inv_label_dict={'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3},
            task_format='classification',
            num_tokens_to_predict=1
        )
    elif dataset == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
        dataset_config = DatasetConfig(
            dataset='trec',
            prompt_prefix="Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n",
            q_prefix="Question: ",
            a_prefix="Answer Type: ",
            label_dict={0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']},
            inv_label_dict={'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5},
            task_format='classification',
            num_tokens_to_predict=1
        )
    elif dataset == 'rte':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()
        dataset_config = DatasetConfig(
            dataset='rte',
            prompt_prefix="",
            q_prefix=" ",
            a_prefix="answer: ",
            label_dict={0: ['False'], 1: ['True']},
            inv_label_dict={'False': 0, 'True': 1},
            task_format='classification',
            num_tokens_to_predict=1
        )
    elif dataset == 'cola':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_cola()
        dataset_config = DatasetConfig(
            dataset='cola',
            prompt_prefix="",
            q_prefix="Sentence: ",
            a_prefix="Hypothesis: the sentence is grammatical, true or false? ",
            label_dict={0: ['false'], 1: ['true']},
            inv_label_dict={'false': 0, 'true': 1},
            task_format='classification',
            num_tokens_to_predict=1
        )
    else:
        raise NotImplementedError
    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels, dataset_config

def random_sampling(sentences, labels, num, is_test=False):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    if is_test:
        idxs_candidate = np.random.choice(len(labels), size=2*num, replace=False)
        idxs = []
        for label in range(max(labels)+1):
            cnt = 0
            for idx in idxs_candidate:
                if label == labels[idx]:
                    idxs.append(idx)
                    cnt += 1
                    if cnt>=num/(max(labels)+1):
                        break
    else:
        idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)

def construct_prompt(
        dataset_config: DatasetConfig, 
        train_sentences, train_labels, test_sentence
    ):
    """construct a single prompt to be fed into the model"""
    prompt, q_prefix, a_prefix = dataset_config.prompt_prefix, dataset_config.q_prefix, dataset_config.a_prefix
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert dataset_config.task_format == "classification"
            l_str = dataset_config.label_dict[l][0] if isinstance(dataset_config.label_dict[l], list) else dataset_config.label_dict[l]
        else:
            assert isinstance(l, str) # string labels
            assert dataset_config.task_format == "qa"
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt