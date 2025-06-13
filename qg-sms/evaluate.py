import re
import os
import tqdm
import time
import json
import openai
import logging
import argparse
import tiktoken
import functools
import multiprocessing
from sklearn.metrics import cohen_kappa_score
from typing import List, Dict, Any, Optional


def load_json(path) :
    with open(path, "r") as fin :
        obj = json.load(fin)
    return obj
def dump_json(obj, path) :
    with open(path, "w") as fout :
        json.dump(obj, fout, indent = 2)


def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    """Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system
    name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho's
    there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'Knock knock.'},
     {'role': 'assistant', 'content': "Who's there?"},
     {'role': 'user', 'content': 'Orange.'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    def string_to_dict(to_convert):
        """Converts a string with equal signs to dictionary. E.g.
        >>> string_to_dict(" name=user university=stanford")
        {'name': 'user', 'university': 'stanford'}
        """
        return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


model_and_tokenizer = {}
def get_modules(eval_step) :
    def process_openai_kwargs(model_name, openai_kwargs : dict) :
        if "gpt-4o" in model_name  or "o1" in model_name:
            tokenizer = tiktoken.get_encoding("o200k_base")
        else:
            tokenizer = tiktoken.encoding_for_model(model_name)
        logit_bias = {}

        if "tokens_to_avoid" in openai_kwargs :
            for t in openai_kwargs["tokens_to_avoid"] :
                curr_tokens = tokenizer.encode(t)
                if len(curr_tokens) != 1 :
                    continue
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = -100  # avoids certain tokens
            openai_kwargs.pop("tokens_to_avoid")

        if "tokens_to_favor" in openai_kwargs :
            for t in openai_kwargs["tokens_to_favor"]:
                curr_tokens = tokenizer.encode(t)
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = 7  # increase log prob of tokens to match
            openai_kwargs.pop("tokens_to_favor")

        if logit_bias :
            openai_kwargs["logit_bias"] = logit_bias
        
        return openai_kwargs

    configs = load_json(os.path.join("config", eval_step + ".json"))
    def get_module(config : dict, eval_step) :
        with open(os.path.join("prompts", config["prompt"]), "r") as fin :
            prompt = fin.read()
        if "Step3" in eval_step:
            return {
                "prompt" : prompt,
                "openai_kwargs" : process_openai_kwargs(config["model_name"], config["openai_kwargs"]),
                "parsing" : config["parsing"],
            }
        else:
            return {
                "prompt" : prompt,
                "openai_kwargs" : process_openai_kwargs(config["model_name"], config["openai_kwargs"]),
            }
    modules = list(map(get_module, configs, [eval_step]))
    return modules


def openai_completion(
    prompt,
    openai_kwargs : dict,
    sleep_time : int = 60,
    request_timeout : int = 30,
) :
    openai_kwargs = openai_kwargs.copy()
    
    while True :
        try :
            completion_batch = openai.ChatCompletion.create(messages = prompt_to_chatml(prompt), request_timeout = request_timeout, **openai_kwargs)

            choice = completion_batch.choices[0]
            assert choice.message.role == "assistant"


            return ("" if "content" not in choice.message else choice.message.content), choice.finish_reason
        except openai.error.OpenAIError as e :
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce" in str(e) :
                openai_kwargs["max_tokens"] = int(openai_kwargs["max_tokens"] * 0.8)
                logging.warning(f"Reducing target length to {openai_kwargs['max_tokens']}, Retrying...")
            elif "content management policy" in str(e) :
                return "", "content management policy", 0.0
            elif "0 is less than the minimum of" in str(e) :
                return "", "0 is less than the minimum of", 0.0
            else :
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    

def Complete(prompt, module) :
    if "openai_kwargs" in module :
        return openai_completion(prompt, module["openai_kwargs"])
    else :
        raise NotImplementedError


def completion_with_parsing(prompt, module) :
    completion, finish_reason = Complete(prompt, module)
    completion = completion.strip()
    parsing : dict = module["parsing"]
    for return_value, exp in parsing.items() :
        if (re.compile(exp)).search(completion) :
            return completion, finish_reason, return_value
    return completion, finish_reason, None


def annotate(instance, modules) :
    def reverse_label(label) :
        if label is None :
            return None
        elif label == "1" :
            return "2"
        elif label == "2" :
            return "1"
        else :
            raise NotImplementedError


    results = []
    final_kwargs = dict(
                        requirement = instance['requirement'],
                        requirement_description = instance['requirement_description']
                        )
    for index, module in enumerate(modules[: -1]) :
        prompt = module["prompt"].format(input = instance["input"])
        auxiliary_input, finish_reason = Complete(prompt, module)
        auxiliary_input = auxiliary_input.strip()
        results.append([auxiliary_input, finish_reason])
        final_kwargs["auxiliary_input_{}".format(index)] = auxiliary_input

    module = modules[-1]
    result = {}
    for swap in (False, True) :
        final_kwargs["question_1"] = instance["question_1"] if not swap else instance["question_2"]
        final_kwargs["question_2"] = instance["question_2"] if not swap else instance["question_1"]
        final_kwargs["predicted_student_performance"] = instance["common_student_answers"] if not swap else instance["common_student_answers_swapped"]

        prompt = module["prompt"].format_map(final_kwargs)
        completion, finish_reason, winner = completion_with_parsing(prompt, module)
        result["swap = {}".format(swap)] = {
            "completion" : [completion, finish_reason],
            "winner" : winner if not swap else reverse_label(winner),
        }
    results.append(result)

    instance["results"] = results
    return instance

def simulate(instance, modules, eval_step) :
    if "Step1" in eval_step:
        final_kwargs = dict(L_full_text = instance["L_full_text"])
    else:
        final_kwargs = dict(L_full_text = instance["learning_mat"],
                        formatted_student_list = instance['formatted_student_list'],
                        question_1 = instance['question_1'],
                        question_2 = instance['question_2']
                        )

    module = modules[-1]
    prompt = module["prompt"].format_map(final_kwargs)

    # --- START OF MODIFICATION ---
    max_retries = 5
    parsed_response = None

    for attempt in range(max_retries):
        completion, finish_reason = Complete(prompt, module)
        try:
            temp_parsed_response = json.loads(completion)
            
            # Initial validation: check if 'response' key exists and its value is a list of dictionaries
            if 'response' in temp_parsed_response and \
               isinstance(temp_parsed_response['response'], list) and \
               all(isinstance(item, dict) for item in temp_parsed_response['response']):

                all_subdicts_valid = True # Flag to track if all sub-dictionaries meet key requirements

                if "Step1" in eval_step:
                    required_keys = {"name", "understanding"}
                    for item in temp_parsed_response['response']:
                        if not required_keys.issubset(item.keys()):
                            logging.warning(f"Attempt {attempt + 1}: Step 1 sub-dictionary missing required keys {required_keys}. Retrying...")
                            all_subdicts_valid = False
                            break # No need to check further items in this response
                elif "Step2" in eval_step:
                    required_keys = {"s_id", "name", "predicted_accuracy_q1", "explanation_q1", "predicted_accuracy_q2", "explanation_q2"}
                    if 'student_list' not in instance:
                        logging.error(f"Error: 'student_list' key missing from instance for Step 2 validation.")
                        all_subdicts_valid = False # Mark as invalid to trigger retry or error out
                    else:
                        # Get a set of s_ids from the original student list for efficient comparison
                        og_s_ids_set = {s['s_id'] for s in instance['student_list'] if 's_id' in s}
                        
                        parsed_s_ids_set = set() # To store s_ids found in the model's response

                        for item in temp_parsed_response['response']:
                            # First, check for all required keys in the sub-dictionary
                            if not required_keys.issubset(item.keys()):
                                logging.warning(f"Attempt {attempt + 1}: Step 2 sub-dictionary missing required keys {required_keys}. Retrying...")
                                all_subdicts_valid = False
                                break # Stop checking items if one is invalid
                            
                            # If s_id exists, add it to our set of parsed s_ids
                            if 's_id' in item:
                                parsed_s_ids_set.add(item['s_id'])
                        
                        # After checking all sub-dictionaries for keys, compare the s_id sets
                        if all_subdicts_valid and og_s_ids_set != parsed_s_ids_set:
                            logging.warning(f"Attempt {attempt + 1}: Step 2 's_id' mismatch. Expected {sorted(list(og_s_ids_set))}, Got {sorted(list(parsed_s_ids_set))}. Retrying...")
                            all_subdicts_valid = False

                if all_subdicts_valid:
                    parsed_response = temp_parsed_response
                    break # Successfully parsed and validated, exit the retry loop
                else:
                    # If sub-dict validation failed, it will go to the outer 'except' or fall through to sleep
                    pass # Logged inside the conditional block above
            else:
                logging.warning(f"Attempt {attempt + 1}: Invalid JSON structure or 'response' key missing/not a list of dicts. Retrying...")
        except json.JSONDecodeError:
            logging.warning(f"Attempt {attempt + 1}: Failed to decode JSON from completion. Retrying...")
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}: An unexpected error occurred during parsing or validation: {e}. Retrying...")

        time.sleep(2 * (attempt + 1)) # Exponential backoff before retrying

    if parsed_response is None:
        logging.error(f"Failed to get a valid 'response' list of dicts after {max_retries} attempts for instance: {instance.get('l_id', 'Unknown ID')}. Returning empty list.")
        # Handle the critical failure by returning an empty list to prevent downstream errors
        if "Step1" in eval_step:
            instance['student_list'] = []
        else:
            instance['student_ans_list_both'] = []
        return instance
    # --- END OF MODIFICATION ---

    if "Step1" in eval_step:
        instance['student_list'] = parsed_response['response'] # Use the parsed_response
        for s_id, student in enumerate(instance['student_list']):
            student['s_id'] = s_id
    else:
        instance['student_ans_list_both'] = parsed_response['response'] # Use the parsed_response
    return instance


def set_api(args) :
    if args.api_type is not None :
        openai.api_type = args.api_type
    if args.api_version is not None :
        openai.api_version = args.api_version
    if args.api_base is not None :
        openai.api_base = args.api_base
    if args.api_key is not None :
        openai.api_key = args.api_key
    if args.organization is not None :
        openai.organization = args.organization
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, required = True)
    parser.add_argument("--num_procs", type = int, default = 10)
    parser.add_argument("--api_type", type = str, default = None)
    parser.add_argument("--api_version", type = str, default = None)
    parser.add_argument("--api_base", type = str, default = None)
    parser.add_argument("--api_key", type = str, default = None)
    parser.add_argument("--organization", type = str, default = None)
    parser.add_argument("--eval", action='store_true', help="Enable evaluation mode.")
    args = parser.parse_args()

    set_api(args)

    pair_dataset = load_json(os.path.join(args.path, "pair_dataset.json"))
    
    
    # Step 1: Generate Student Profiles
    all_Ls = []
    found_ids = []
    for pair in pair_dataset:
        if pair['l_id'] not in found_ids:
            all_Ls.append({
                'l_id': pair['l_id'],
                'L_full_text': pair['learning_mat'],
            })
            found_ids.append(pair['l_id'])
    logging.info(f"Number of unique learning material: {len(all_Ls)}")
    
    modules = get_modules("Step1_ProfileGeneration")
    with multiprocessing.Pool(args.num_procs) as p :
        _simulate = functools.partial(simulate, modules = modules, eval_step = "Step1_ProfileGeneration")
        all_Ls = list(
            tqdm.tqdm(
                p.imap(_simulate, all_Ls),
                desc = "Generating Student Profiles for each learning Learning Material",
                total = len(all_Ls),
            )
        )
        
    dump_json(
            all_Ls,
            os.path.join(args.path, "gen_students_by_L.json")
    )
    
    # Step 2: Predict Student Performance
    
    def find_dict_by_id(id_field, target_id: Any, list_of_dicts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Finds and returns the first dictionary in a list that has a matching 'id' key.

        Args:
            target_id: The ID to search for. Can be any type (int, str, etc.) that
                    matches the type of the 'id' values in the dictionaries.
            list_of_dicts: A list where each element is a dictionary expected to
                        have an 'id' key.

        Returns:
            The dictionary with the matching 'id', or None if no dictionary with
            that id is found.
        """
        for d in list_of_dicts:
            # It's good practice to check if 'id' key exists before accessing it
            if id_field in d and d[id_field] == target_id:
                return d
        return None
    
    
    def format_student_profiles(student_list):
        student_profiles = ""
        for student in student_list:
            student_profiles += f"Student ID: {student['s_id']}\n"
            student_profiles += f"Name: {student['name']}\n"
            student_profiles += f"Understanding: {student['understanding']}\n\n"

        return student_profiles
    
    for pair in pair_dataset:
        l_id = pair['l_id']
        student_list = find_dict_by_id('l_id', l_id, all_Ls)['student_list']
        pair['student_list'] = student_list
        pair['formatted_student_list'] = format_student_profiles(student_list)
        
    modules = get_modules("Step2_PerformancePrediction")
    with multiprocessing.Pool(args.num_procs) as p :
        _simulate = functools.partial(simulate, modules = modules, eval_step = "Step2_PerformancePrediction")
        pair_dataset = list(
            tqdm.tqdm(
                p.imap(_simulate, pair_dataset),
                desc = "Predicting Student Performance for each pair",
                total = len(pair_dataset),
            )
        )
    
    # Step 3: Evaluation
    def format_student_ans_3(student_list, student_ans_list, swapped=False):
        student_profiles = ""
        for student in student_list:
            try:
                student_profiles += f"Student ID: {student['s_id']}\n"
                student_profiles += f"Name: {student['name']}\n"
                student_profiles += f"Student's Actual Understanding of Lecture: {student['understanding']}\n"
                # student_profiles += f"Student's Overall Performance: {students_understanding_dict[(lec_id, student['student_id'])][1]}\n"
                if not swapped:
                    student_profiles += f"Student's Performance on Output (a): {find_dict_by_id('s_id', student['s_id'], student_ans_list)['predicted_accuracy_q1']}\n"
                    student_profiles += f"Explantion for Student's Performance: {find_dict_by_id('s_id', student['s_id'], student_ans_list)['explanation_q1']}\n"
                    student_profiles += f"Student's Performance on Output (b): {find_dict_by_id('s_id', student['s_id'], student_ans_list)['predicted_accuracy_q2']}\n"
                    student_profiles += f"Explantion for Student's Performance: {find_dict_by_id('s_id', student['s_id'], student_ans_list)['explanation_q2']}\n\n"
                else:
                    student_profiles += f"Student's Performance on Output (a): {find_dict_by_id('s_id', student['s_id'], student_ans_list)['predicted_accuracy_q2']}\n"
                    student_profiles += f"Explantion for Student's Performance: {find_dict_by_id('s_id', student['s_id'], student_ans_list)['explanation_q2']}\n"
                    student_profiles += f"Student's Performance on Output (b): {find_dict_by_id('s_id', student['s_id'], student_ans_list)['predicted_accuracy_q1']}\n"
                    student_profiles += f"Explantion for Student's Performance: {find_dict_by_id('s_id', student['s_id'], student_ans_list)['explanation_q1']}\n\n"
            except:
                pass
        return student_profiles

    for pair in pair_dataset:
        l_id = pair['l_id']
        pair['common_student_answers'] = format_student_ans_3(find_dict_by_id('l_id', l_id, all_Ls)['student_list'], pair['student_ans_list_both'],  False)
        pair['common_student_answers_swapped'] = format_student_ans_3(find_dict_by_id('l_id', l_id, all_Ls)['student_list'], pair['student_ans_list_both'],  True)
       
    modules = get_modules("Step3_Evaluation")
    with multiprocessing.Pool(args.num_procs) as p :
        _annotate = functools.partial(annotate, modules = modules)
        pair_dataset = list(
            tqdm.tqdm(
                p.imap(_annotate, pair_dataset),
                desc = "Evaluatig question pairs using Simulated Students",
                total = len(pair_dataset),
            )
        )
   
    
    dataset = pair_dataset
    dump_json(dataset, os.path.join(args.path, "eval_result.json"))
    if args.eval:
        correct_False, correct_True, correct_both, equal = 0, 0, 0, 0
        kappa_y1_False, kappa_y1_True, kappa_y2 = [], [], []
        for instance in dataset :
        
                    
                
            label = str(instance["label"])
            output_False = instance["results"][-1]["swap = False"]["winner"]
            output_True  = instance["results"][-1]["swap = True"]["winner"]
            
            

            correct_False += (output_False == label)
            correct_True += (output_True == label)
            correct_both += ((output_False == label) and (output_True == label))
            equal += (output_True == output_False)

            label = instance["label"]
            output_False = int(output_False) if output_False in ("0", "1") else 0
            output_True = int(output_True) if output_True in ("0", "1") else 0
            kappa_y1_False.append(output_False)
            kappa_y1_True.append(output_True)
            kappa_y2.append(label)
    
        statistics = {
            "correct_False" : "{} / {} = {}%".format(correct_False, len(dataset), correct_False / len(dataset) * 100),
            "correct_True" : "{} / {} = {}%".format(correct_True, len(dataset), correct_True / len(dataset) * 100),
            "correct_average" : "{}%".format((correct_False + correct_True) / 2 / len(dataset) * 100),
            "correct_both" : "{} / {} = {}%".format(correct_both, len(dataset), correct_both / len(dataset) * 100),
            "equal" : "{} / {} = {}%".format(equal, len(dataset), equal / len(dataset) * 100),
        }
        statistics["kappa_False"] = cohen_kappa_score(kappa_y1_False, kappa_y2)
        statistics["kappa_True"] = cohen_kappa_score(kappa_y1_True, kappa_y2)
        statistics["kappa_average"] = (statistics["kappa_False"] + statistics["kappa_True"]) / 2
        statistics["kappa_agreement"] = cohen_kappa_score(kappa_y1_False, kappa_y1_True)
        dump_json(
            statistics,
            os.path.join(args.path, "statistics.json")
        )
        
if __name__ == "__main__" :
    main()