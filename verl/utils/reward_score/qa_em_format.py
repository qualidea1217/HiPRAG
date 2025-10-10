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

import asyncio
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
import re
import string
import random
import torch
import logging
from typing import Union, List
from verl import DataProto
from tensordict import TensorDict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem, collate_fn

logger = logging.getLogger(__name__)

client = OpenAI(api_key="")
client_async = AsyncOpenAI(api_key="")
# client = AzureOpenAI(
#     azure_endpoint="",
#     api_key="",
#     api_version=""
# )
# client_async = AsyncAzureOpenAI(
#     azure_endpoint="",
#     api_key="",
#     api_version=""
# )

SEARCH_STEP_VERIFY_PROMPT_V1 = "You are an expert in Natural Language Understanding and Semantic Analysis. Your goal is to determine if these two statements are semantically equivalent—that is, if they mean the same thing and convey the same core information. Provide your answers with a single boolean value \"True\" or \"False\" in the tag <answer></answer> (e.g. <answer>True</answer> or <answer>False</answer>)."

NON_SEARCH_STEP_VERIFY_PROMPT_V1 = """You are an expert Fact-Checker and Logic Verifier. Your task is to evaluate a single, isolated reasoning step from an AI agent.

This step was generated without using a search tool. Your goal is to determine if the agent made a mistake by not searching, based only on the information within this single step and your own general knowledge.

Analyze the provided step by asking two questions:
1. Factual Accuracy: Is the statement in the <reasoning></reasoning> and <conclusion></conclusion> factually correct?
2. Internal Logic: Does the <conclusion></conclusion> logically follow from the <reasoning></reasoning> provided within this same step?

If both questions are answered correctly, provide your answers with a single boolean value "True" or "False" in the tag <answer></answer> (e.g. <answer>True</answer> or <answer>False</answer>).
"""

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


async def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0.2, final_format_score=0, retrieval_score=0, format_score=0, score=1., actor_rollout_wg=None, tokenizer=None, all_search_query_dict=None):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # is_valid_format, _ = is_valid_sequence(solution_str)
    # retrieval_correct = False
    # if is_valid_format:
    #     retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    # try:
    print(f"--------------------------------")
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if structure_format_score == 0:  # only for testing and validation
        final_reward = 1 if cover_exact_match(answer, ground_truth['target']) else 0
    else:
        final_reward = search_r1_format(solution_str, ground_truth['target'], lambda_f=structure_format_score)

    # do over-search and under-search evaluation sync version
    # if final_reward == 1:
    #     correct_count = 0
    #     step_content_list, _ = get_steps_from_full_response(solution_str)
    #     for step_content in step_content_list:
    #         if "<search>" in step_content:  # search step
    #             is_oversearched = test_oversearch_direct(step_content, tokenizer, client, all_search_query_dict, actor_rollout_wg)
    #             if not is_oversearched:
    #                 correct_count += 1
    #             print(f"Search step is {'over' if is_oversearched else 'not over'}-searched")
    #         else:  # non-search step
    #             is_undersearched = test_undersearch(step_content, client)
    #             if not is_undersearched:
    #                 correct_count += 1
    #             print(f"Non-search step is {'under' if is_undersearched else 'not under'}-searched")
    #     final_reward += 0.4 * correct_count / len(step_content_list)

    # do over-search and under-search evaluation using async openai client
    # --- ASYNC over/under-search evaluation ---

    if final_reward == 1 and structure_format_score != 0:
        correct_count = 0
        step_content_list, _ = get_steps_from_full_response(solution_str)

        # async def _evaluate_steps_async():
        #     tasks = []
        #     for step_content in step_content_list:
        #         if "<search>" in step_content:
        #             tasks.append(
        #                 test_oversearch_direct_async(
        #                     step_content=step_content,
        #                     tokenizer=tokenizer,
        #                     client=client_async,
        #                     all_search_query_dict=all_search_query_dict,
        #                     actor_rollout_wg=actor_rollout_wg,
        #                 )
        #             )
        #         else:
        #             tasks.append(
        #                 test_undersearch_async(
        #                     step_content=step_content,
        #                     gpt_client=client_async,
        #                 )
        #             )
        #     return await asyncio.gather(*tasks)

        # results = asyncio.run(_evaluate_steps_async())

        # for step_content, res in zip(step_content_list, results):
        #     if "<search>" in step_content:
        #         is_oversearched = res
        #         if not is_oversearched:
        #             correct_count += 1
        #         print(f"Search step is {'over' if is_oversearched else 'not over'}-searched")
        #     else:
        #         is_undersearched = res
        #         if not is_undersearched:
        #             correct_count += 1
        #         print(f"Non-search step is {'under' if is_undersearched else 'not under'}-searched")

        total_count = 0
        for step_content in step_content_list:
            if "<search>" in step_content:
                total_count += 1
                is_oversearched = await test_oversearch_direct_async(step_content, tokenizer, client_async, all_search_query_dict, actor_rollout_wg)
                if not is_oversearched:
                    correct_count += 1
                print(f"Search step is {'over' if is_oversearched else 'not over'}-searched")
            else:
                is_undersearched = await test_undersearch_async(step_content, client_async)
                if not is_undersearched:
                    correct_count += 1
                print(f"Non-search step is {'under' if is_undersearched else 'not under'}-searched")

        if step_content_list:
            final_reward += (0.4 * correct_count / total_count) if total_count > 0 else 0.4
    # --- END OF ASYNC over/under-search evaluation ---
    
    if do_print:
        print(f"--------------------------------")
        if structure_format_score == 0:
            print("Test Sample")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Final reward: {final_reward}")
        print(f"Solution string: {solution_str}")
    # except Exception as e:
    #     print(f"Error: {e}")
    #     final_reward = 0
    #     print(f"Solution string: {solution_str}")
    #     print(f"Solution String failed to be processed, directly set final reward to {final_reward}")
            
    # if answer is None:
    #     if is_valid_format:
    #         if retrieval_correct:
    #             return structure_format_score + retrieval_score # 0.3
    #         else:
    #             return structure_format_score # 0.2
    #     else:
    #         return 0
    # else:
    #     if em_check(answer, ground_truth['target']):
    #         if is_valid_format:
    #             return score # 1
    #         else:
    #             return score - structure_format_score # 0.8
    #     elif is_valid_format:
    #         if retrieval_correct:
    #             return structure_format_score + retrieval_score # 0.3
    #         else:
    #             return structure_format_score # 0.2
    #     else:
    #         return final_format_score # 0.1
    return final_reward


def normalize_string(str):
    """Normalize answer by removing articles, punctuation, extra spaces, and lowercasing."""
    # Remove articles, punctuation, lowercase, and fix whitespace in one pass
    str = re.sub(r'\b(a|an|the)\b', ' ', str.lower())
    str = ''.join(c if c not in string.punctuation else ' ' for c in str)
    return ' '.join(str.split())


def cover_exact_match(pred, gold):
    """Check if any golden answer is a substring of the prediction."""
    if not pred:
        return False
    if isinstance(gold, str):
        gold = [gold]
    norm_pred = normalize_string(pred)
    return any(normalize_string(ans) in norm_pred for ans in gold)


def extract_str_between(text, start, end):
    """Extract strings between start and end markers."""
    results = []
    start_idx = 0
    while True:
        start_idx = text.find(start, start_idx)
        if start_idx == -1:
            break
        start_idx += len(start)
        end_idx = text.find(end, start_idx)
        if end_idx == -1:
            break
        results.append(text[start_idx:end_idx])
        start_idx = end_idx + len(end)
    return results


def check_full_response_format(full_response: str) -> bool:
    """
    Check if the model response is in the correct format as specified in AGENT_PROMPT_V2 for agentic rag LLM.
    Returns True if the format is correct, otherwise False.
    Accepts two step formats:
      1. <step><reasoning>...</reasoning><search>...</search><context>...</context><conclusion>...</conclusion></step>
      2. <step><reasoning>...</reasoning><conclusion>...</conclusion></step>
    """
    # Find the last occurrence of <think>
    idx = full_response.rfind('<think>')
    if idx == -1:
        return False
    response = full_response[idx:].strip()
    # Regex for the overall structure: <think>...</think><answer>...</answer>
    pattern = re.compile(
        r'^<think>\s*((<step>.*?</step>\s*)+)</think>\s*<answer>.*?</answer>\s*$',
        re.DOTALL
    )
    match = pattern.match(response)
    if not match:
        return False
    # Extract all <step> blocks
    steps = re.findall(r'<step>(.*?)</step>', match.group(1), re.DOTALL)
    if not steps:
        return False
    # Check each <step> for required tags in order, no duplicates, and valid format
    for step in steps:
        # Find all tags in order
        tags = re.findall(r'<(reasoning|search|context|conclusion)>', step)
        # Acceptable tag sequences:
        # 1. ['reasoning', 'search', 'context', 'conclusion']
        # 2. ['reasoning', 'conclusion']
        if tags == ['reasoning', 'search', 'context', 'conclusion']:
            # Ensure each tag appears only once
            for tag in ['reasoning', 'search', 'context', 'conclusion']:
                if step.count(f'<{tag}>') != 1 or step.count(f'</{tag}>') != 1:
                    return False
            # Ensure <search> and <context> are non-empty
            for tag in ['search', 'context']:
                content = re.search(f'<{tag}>(.*?)</{tag}>', step, re.DOTALL)
                if content is None or content.group(1).strip() == '':
                    return False
            # Ensure no extra tags between <step> and </step> by removing the four required tags and their content
            cleaned = step
            for tag in ['reasoning', 'search', 'context', 'conclusion']:
                cleaned = re.sub(f'<{tag}>.*?</{tag}>', '', cleaned, flags=re.DOTALL)
            if re.search(r'</?[^>]+>', cleaned):
                return False
        elif tags == ['reasoning', 'conclusion']:
            # Ensure each tag appears only once
            for tag in ['reasoning', 'conclusion']:
                if step.count(f'<{tag}>') != 1 or step.count(f'</{tag}>') != 1:
                    return False
            # Ensure no extra tags between <step> and </step> by removing the two required tags and their content
            cleaned = step
            for tag in ['reasoning', 'conclusion']:
                cleaned = re.sub(f'<{tag}>.*?</{tag}>', '', cleaned, flags=re.DOTALL)
            if re.search(r'</?[^>]+>', cleaned):
                return False
        else:
            return False
    return True


def check_full_response_format_optimized(full_response: str) -> bool:
    """
    Optimized version of check_full_response_format.
    Check if the model response is in the correct format as specified in AGENT_PROMPT_V2 for agentic rag LLM.
    Returns True if the format is correct, otherwise False.
    Accepts two step formats:
      1. <step><reasoning>...</reasoning><search>...</search><context>...</context><conclusion>...</conclusion></step>
      2. <step><reasoning>...</reasoning><conclusion>...</conclusion></step>
    """
    # Find the last occurrence of <think>
    idx = full_response.rfind('<think>')
    if idx == -1:
        return False
    
    # Extract the relevant part and strip whitespace
    response = full_response[idx:].strip()
    
    # Quick checks for required structure using string methods instead of regex
    if not response.startswith('<think>'):
        return False
    
    # Find the end of think tag
    think_end = response.find('</think>')
    if think_end == -1:
        return False
    
    # Check for answer tag after think - must handle whitespace
    answer_section = response[think_end + 8:].lstrip()  # Skip '</think>' and whitespace
    if not answer_section.startswith('<answer>'):
        return False
    
    answer_end = answer_section.find('</answer>')
    if answer_end == -1:
        return False
    
    # Check no extra content after </answer>
    remaining = answer_section[answer_end + 9:].strip()
    if remaining:
        return False
    
    # Extract think content (handle potential whitespace)
    think_start = 7  # Length of '<think>'
    # Skip whitespace after <think>
    while think_start < think_end and response[think_start].isspace():
        think_start += 1
    
    think_content = response[think_start:think_end].rstrip()
    
    # Check if there's at least one step
    if '<step>' not in think_content or '</step>' not in think_content:
        return False
    
    # Process each step
    step_start = 0
    while True:
        # Find next step
        step_tag_start = think_content.find('<step>', step_start)
        if step_tag_start == -1:
            break
        
        step_tag_end = think_content.find('</step>', step_tag_start)
        if step_tag_end == -1:
            return False
        
        # Extract step content
        step_content = think_content[step_tag_start + 6:step_tag_end]
        
        # Check step format using a more efficient approach
        if not validate_step_optimized(step_content):
            return False
        
        step_start = step_tag_end + 7  # Move past '</step>'
    
    return True


def validate_step_optimized(step_content: str) -> bool:
    """
    Optimized validation of a single step's content.
    Returns True if the step follows one of the two valid formats.
    """
    # Quick check for reasoning tag (required in both formats)
    reasoning_start = step_content.find('<reasoning>')
    if reasoning_start == -1:
        return False
    
    reasoning_end = step_content.find('</reasoning>', reasoning_start)
    if reasoning_end == -1:
        return False
    
    # Check what comes after reasoning
    after_reasoning = step_content[reasoning_end + 12:]
    
    # Check for search tag (format 1)
    if '<search>' in after_reasoning:
        # Format 1: reasoning, search, context, conclusion
        search_start = after_reasoning.find('<search>')
        search_end = after_reasoning.find('</search>', search_start)
        if search_end == -1:
            return False
        
        # Check if search is non-empty
        search_content = after_reasoning[search_start + 8:search_end].strip()
        if not search_content:
            return False
        
        # Check for context
        after_search = after_reasoning[search_end + 9:]
        context_start = after_search.find('<context>')
        if context_start == -1:
            return False
        
        context_end = after_search.find('</context>', context_start)
        if context_end == -1:
            return False
        
        # Check if context is non-empty
        context_content = after_search[context_start + 9:context_end].strip()
        if not context_content:
            return False
        
        # Check for conclusion
        after_context = after_search[context_end + 10:]
        conclusion_start = after_context.find('<conclusion>')
        if conclusion_start == -1:
            return False
        
        conclusion_end = after_context.find('</conclusion>', conclusion_start)
        if conclusion_end == -1:
            return False
        
        # Verify tag counts using a more efficient method
        tag_counts = count_tags_optimized(step_content)
        if (tag_counts.get('reasoning', 0) != 1 or
            tag_counts.get('search', 0) != 1 or
            tag_counts.get('context', 0) != 1 or
            tag_counts.get('conclusion', 0) != 1):
            return False
        
        # Check for extra tags
        if has_extra_tags_format1(step_content):
            return False
        
    else:
        # Format 2: reasoning, conclusion
        conclusion_start = after_reasoning.find('<conclusion>')
        if conclusion_start == -1:
            return False
        
        conclusion_end = after_reasoning.find('</conclusion>', conclusion_start)
        if conclusion_end == -1:
            return False
        
        # Verify tag counts
        tag_counts = count_tags_optimized(step_content)
        if (tag_counts.get('reasoning', 0) != 1 or
            tag_counts.get('conclusion', 0) != 1):
            return False
        
        # Check for extra tags
        if has_extra_tags_format2(step_content):
            return False
    
    return True


def count_tags_optimized(content: str) -> dict:
    """
    Efficiently count occurrences of specific tags.
    """
    tags = ['reasoning', 'search', 'context', 'conclusion']
    counts = {}
    
    for tag in tags:
        open_tag = f'<{tag}>'
        close_tag = f'</{tag}>'
        
        # Count occurrences
        open_count = content.count(open_tag)
        close_count = content.count(close_tag)
        
        if open_count > 0 or close_count > 0:
            counts[tag] = open_count
    
    return counts


def has_extra_tags_format1(content: str) -> bool:
    """
    Check for extra tags in format 1 (with all 4 tags).
    Optimized version using string operations instead of regex.
    """
    # Create a mutable copy of content
    cleaned = content
    
    # Remove all valid tag pairs and their content
    for tag in ['reasoning', 'search', 'context', 'conclusion']:
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'
        
        while True:
            start_idx = cleaned.find(start_tag)
            if start_idx == -1:
                break
            
            end_idx = cleaned.find(end_tag, start_idx)
            if end_idx == -1:
                break
            
            # Remove the tag and its content
            cleaned = cleaned[:start_idx] + cleaned[end_idx + len(end_tag):]
    
    # Check if any tags remain by looking for < and > characters
    return '<' in cleaned and '>' in cleaned


def has_extra_tags_format2(content: str) -> bool:
    """
    Check for extra tags in format 2 (only reasoning and conclusion).
    Optimized version using string operations instead of regex.
    """
    # Create a mutable copy of content
    cleaned = content
    
    # Remove all valid tag pairs and their content
    for tag in ['reasoning', 'conclusion']:
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'
        
        while True:
            start_idx = cleaned.find(start_tag)
            if start_idx == -1:
                break
            
            end_idx = cleaned.find(end_tag, start_idx)
            if end_idx == -1:
                break
            
            # Remove the tag and its content
            cleaned = cleaned[:start_idx] + cleaned[end_idx + len(end_tag):]
    
    # Check if any tags remain by looking for < and > characters
    return '<' in cleaned and '>' in cleaned


def search_r1_format(model_response, ground_truth_answer, lambda_f=0.4):
    """
    Implements the reward function of Search R1 from Equation 3 in Section 4.1 in the paper "An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents" (https://arxiv.org/abs/2505.15117).
    
    Args:
        model_response: The complete model response string
        ground_truth_answer: The correct answer (string or list of strings)
        lambda_f: Format reward weight (default 0.4 for 7B models)
    
    Returns:
        float: Reward score between 0 and 1
    """
    # Extract predicted answer
    answer_matches = extract_str_between(model_response, '<answer>', '</answer>')
    predicted_answer = answer_matches[-1].strip() if answer_matches else ""
    
    # Check correctness and format
    answer_correct = cover_exact_match(predicted_answer, ground_truth_answer)
    format_correct = check_full_response_format_optimized(model_response)
    
    # Apply Equation 3
    if answer_correct:
        return 1.0 if format_correct else 1.0 - lambda_f
    else:
        return lambda_f if format_correct else 0.0


def get_steps_from_full_response(full_response: str) -> tuple[list[str], list[int]]:
    """
    Get the step content and start index of each step from the full response.
    """
    step_content_list, step_start_list = [], []
    think_start = full_response.rfind("<think>")
    think_end = full_response.find("</think>", think_start)
    step_start = think_start
    while step_start < think_end:
        # Find next step
        step_start = full_response.find('<step>', step_start)
        if step_start == -1:
            break
        step_end = full_response.find('</step>', step_start)
        if step_end == -1:
            break
        # Extract step content (excluding <step> and </step> tags)
        step_content = full_response[step_start + 6:step_end]
        step_content_list.append(step_content)
        step_start_list.append(step_start)
        step_start = step_end + 7  # Move past '</step>'
    return step_content_list, step_start_list


def test_oversearch_direct(step_content: str, tokenizer, client: OpenAI, all_search_query_dict: dict, actor_rollout_wg=None) -> bool:
    """
    Test the oversearch behavior of a single step by directly asking the search query to the model.
    """
    # Extract the search query and original conclusion of this step
    search_query = extract_str_between(step_content, "<search>", "</search>")[-1]
    conclusion_content = extract_str_between(step_content, "<conclusion>", "</conclusion>")[-1]

    # Generate response by directly asking the search query to the model
    try:
        non_search_output = all_search_query_dict[search_query]
    except KeyError:
        # if tokenizer.chat_template:
        #     messages = [{"role": "user", "content": f"Please answer the following question or provide relevant information to the statement: {search_query}"}]
        #     prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # else:
        #     prompt = search_query

        # if actor_rollout_wg and tokenizer:
        #     input_prompts, outputs = generate_extra_outputs_for_qa_em(
        #         input_strings=[prompt],
        #         actor_rollout_wg=actor_rollout_wg,
        #         tokenizer=tokenizer,
        #         temperature=1.0,
        #         max_prompt_length=512
        #     )
            
        #     # 打印结果
        #     non_search_output = outputs[0]
        #     print(f"Extra Generation:Prompt: {input_prompts[0]!r}, Generated text: {non_search_output!r}")
        # else:
        #     non_search_output = ""
        return False

    # Determine if the two conclusions are equivalent
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStatement 1: {conclusion_content}\n\nStatement 2: {non_search_output}"}]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
        return True if "True" in is_correct else False
    except Exception as e:
        print(f"Error in test_oversearch_direct: {e}")
        return False


async def test_oversearch_direct_async(step_content: str, tokenizer, client: AsyncOpenAI, all_search_query_dict: dict, actor_rollout_wg=None) -> bool:
    """
    Test the oversearch behavior of a single step by directly asking the search query to the model.
    """
    # Extract the search query and original conclusion of this step
    try:
        search_query = extract_str_between(step_content, "<search>", "</search>")[-1]
        conclusion_content = extract_str_between(step_content, "<conclusion>", "</conclusion>")[-1]
    except Exception as e:
        print(f"Error in test_oversearch_direct_async (getting search query or conclusion): {e}")
        return False

    # Generate response by directly asking the search query to the model
    try:
        non_search_output = all_search_query_dict[search_query]
    except Exception as e:
        # if tokenizer.chat_template:
        #     messages = [{"role": "user", "content": f"Please answer the following question or provide relevant information to the statement: {search_query}"}]
        #     prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # else:
        #     prompt = search_query

        # if actor_rollout_wg and tokenizer:
        #     input_prompts, outputs = generate_extra_outputs_for_qa_em(
        #         input_strings=[prompt],
        #         actor_rollout_wg=actor_rollout_wg,
        #         tokenizer=tokenizer,
        #         temperature=1.0,
        #         max_prompt_length=512
        #     )
            
        #     # 打印结果
        #     non_search_output = outputs[0]
        #     print(f"Extra Generation:Prompt: {input_prompts[0]!r}, Generated text: {non_search_output!r}")
        # else:
        #     non_search_output = ""
        print(f"Error in test_oversearch_direct_async (getting non-search output): {e}")
        return False

    # Determine if the two conclusions are equivalent
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStatement 1: {conclusion_content}\n\nStatement 2: {non_search_output}"}]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
        return True if "True" in is_correct else False
    except Exception as e:
        print(f"Error in test_oversearch_direct_async: {e}")
        return False


def test_undersearch(step_content: str, client: OpenAI) -> bool:
    """
    Test the undersearch behavior of a single step using openai client.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": NON_SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStep Content: {step_content}"}]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
        return False if "True" in is_correct else True
    except Exception as e:
        print(f"Error in test_undersearch: {e}")
        return False


async def test_undersearch_async(step_content: str, gpt_client: AsyncOpenAI) -> bool:
    """
    Test the undersearch behavior of a single step using async openai client.
    """
    try:
        completion = await gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": NON_SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStep Content: {step_content}"}]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
        return False if "True" in is_correct else True
    except Exception as e:
        print(f"Error in test_undersearch_async: {e}")
        return False


def generate_extra_outputs_for_qa_em(
    input_strings: Union[str, List[str]],
    actor_rollout_wg,
    tokenizer,
    max_prompt_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    batch_size: int = 128,
    **kwargs
) -> Union[str, List[str]]:
    """
    专门为qa_em_format设计的生成函数
    可以直接在compute_score_em中使用
    
    Args:
        input_strings: 输入字符串或字符串列表
        actor_rollout_wg: RayWorkerGroup实例
        tokenizer: 分词器
        max_prompt_length: 最大提示长度
        temperature: 采样温度
        top_p: nucleus采样参数
        top_k: top-k采样参数
        batch_size: 批处理大小
        **kwargs: 其他参数
    
    Returns:
        生成的输出字符串或字符串列表
    """
    
    # 输入预处理
    if isinstance(input_strings, str):
        input_list = [input_strings]
        return_single = True
    else:
        input_list = input_strings
        return_single = False
    
    if not input_list:
        return "" if return_single else []
    
    # try:
        # 分批处理
    all_inputs = []
    all_outputs = []
    
    for i in range(0, len(input_list), batch_size):
        batch_inputs = input_list[i:i + batch_size]
        
        # 构建生成batch
        gen_batch = _prepare_qa_generation_batch(
            batch_inputs, tokenizer, actor_rollout_wg.world_size, max_prompt_length
        )
        
        # 设置采样参数
        sampling_kwargs = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            **kwargs
        }
        
        # 调用生成方法（对齐到 world_size 的等分）
        if hasattr(actor_rollout_wg, 'world_size'):
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, actor_rollout_wg.world_size)
        else:
            gen_batch_padded, pad_size = gen_batch, 0

        gen_batch_padded, pad_size = ensure_batch_is_padded(gen_batch_padded, gen_batch, actor_rollout_wg.world_size, pad_size)
        gen_output_padded = actor_rollout_wg.generate_sequences(gen_batch_padded)
        gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size) if pad_size else gen_output_padded
        # gen_output = _generate_with_gpu_padding(gen_batch, actor_rollout_wg)
        
        # 解码输入与输出
        batch_inputs, batch_outputs = _decode_qa_generation_output(gen_output, tokenizer)
        all_inputs.extend(batch_inputs)
        all_outputs.extend(batch_outputs)
    
    # 返回结果
    # 始终返回两个列表，索引一一对应
    return all_inputs, all_outputs
        
    # except Exception as e:
    #     logger.error(f"QA generation failed: {e}")
    #     # 返回空结果而不是抛出异常
    #     return "" if return_single else [""] * len(input_list)


def _prepare_qa_generation_batch(
    prompts_text: List[str], 
    tokenizer, 
    size_divisor: int = None,
    max_prompt_length: int = None
) -> DataProto:
    """准备QA生成batch"""
    
    # temporarily change the padding side to left
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch_size = len(prompts_text)

        # pad the batch to the size_divisor
    if size_divisor and batch_size % size_divisor != 0:
        need_to_pad = size_divisor - batch_size % size_divisor
        prompts_text = prompts_text + [prompts_text[0]] * need_to_pad
        batch_size = len(prompts_text)
    
    # 分词和编码
    encoded = tokenizer(
        prompts_text,
        add_special_tokens=False,
        # padding=True,
        # truncation=True,
        # max_length=max_prompt_length,
        padding="longest",
        return_tensors="pt"
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # 构建position_ids
    position_ids = torch.zeros_like(input_ids)
    for i in range(batch_size):
        non_pad_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            first_token_pos = non_pad_indices[0]
            seq_len = attention_mask[i].sum()
            position_ids[i, first_token_pos:first_token_pos + seq_len] = torch.arange(seq_len)
    
    # 构建TensorDict
    batch = TensorDict({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    }, batch_size=batch_size)
    
    # 设置元信息
    meta_info = {
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'do_sample': True,
        'recompute_log_prob': False,
    }

    # restore the padding side
    tokenizer.padding_side = original_padding_side
    
    return DataProto(batch=batch, meta_info=meta_info)


def _decode_qa_generation_output(gen_output: DataProto, tokenizer) -> List[str]:
    """解码QA生成的输入与输出"""
    inputs, outputs = [], []

    if isinstance(gen_output, DataProtoItem):
        # The item itself contains a batch of data. We need to iterate over it.
        batch_td = gen_output.batch
        batch_size = batch_td.batch_size[0] # Get the batch size from the TensorDict

        for i in range(batch_size):
            # Slice the batch TensorDict to get the data for just the i-th item.
            item_td = batch_td[i]

            # Now, all tensors are 1D, and the logic will work correctly.
            prompt_ids = item_td['prompts'] 
            prompt_length = prompt_ids.shape[-1]
            attention_mask = item_td['attention_mask'] 

            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = item_td['responses'] 
            valid_response_length = attention_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Concatenation now works because it's joining two 1D tensors.
            # print(f"valid_prompt_ids: {valid_prompt_ids.shape}, valid_response_ids: {valid_response_ids.shape}")
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))

            sequences_str = tokenizer.decode(sequences)
            # print(f"Got DataProtoItem, sequences_str: {sequences_str}")
            # print(f"input str: {tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)}")
            # print(f"output str: {tokenizer.decode(valid_response_ids, skip_special_tokens=False)}")
            
            # Append the decoded strings to your lists
            inputs.append(tokenizer.decode(valid_prompt_ids, skip_special_tokens=False))
            outputs.append(tokenizer.decode(valid_response_ids, skip_special_tokens=False))
        # data_item = gen_output  # DataProtoItem

        # prompt_ids = data_item.batch['prompts']

        # prompt_length = prompt_ids.shape[-1]

        # valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        # response_ids = data_item.batch['responses']
        # valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        # valid_response_ids = response_ids[:valid_response_length]

        # # decode
        # print(f"valid_prompt_ids: {valid_prompt_ids.shape}, valid_response_ids: {valid_response_ids.shape}")
        # sequences = torch.cat((valid_prompt_ids, valid_response_ids), dim=0)
        # sequences_str = tokenizer.decode(sequences)
        # print(f"Got DataProtoItem, sequences_str: {sequences_str}")
        # print(f"input str: {tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)}")
        # print(f"output str: {tokenizer.decode(valid_response_ids, skip_special_tokens=False)}")
        # inputs.append(tokenizer.decode(valid_prompt_ids, skip_special_tokens=False))
        # outputs.append(tokenizer.decode(valid_response_ids, skip_special_tokens=False))

    else:
        batch_size = len(gen_output)
        
        for i in range(batch_size):
            # 获取输入与响应部分
            data_item = gen_output[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = tokenizer.decode(sequences)
            # print(f"Got DataProto, sequences_str: {sequences_str}")
            # print(f"input str: {tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)}")
            # print(f"output str: {tokenizer.decode(valid_response_ids, skip_special_tokens=False)}")
            inputs.append(tokenizer.decode(valid_prompt_ids, skip_special_tokens=False))
            outputs.append(tokenizer.decode(valid_response_ids, skip_special_tokens=False))
    
    return inputs, outputs


def ensure_batch_is_padded(padded_batch, original_batch, divisor, original_pad_size):
    """
    Checks if a batch is correctly padded. If not, it applies a robust
    padding fix and returns the corrected batch and the new pad_size.
    """
    # Check if the batch size is divisible by the number of workers.
    if len(padded_batch) % divisor != 0:
        print(
            f"⚠️ Original padding failed! (Size: {len(padded_batch)}, Divisor: {divisor}). "
            f"Applying a robust fix..."
        )
        original_size = len(original_batch)
        if original_size == 0:
            return original_batch, 0 # Return empty batch if original was empty

        # --- Apply the robust padding logic ---
        # Calculate the correct number of items to add.
        new_pad_size = divisor - (original_size % divisor)
        
        # Create the padding data by repeating items from the original batch.
        padding_items = [original_batch[i % original_size] for i in range(new_pad_size)]
        padding_data = collate_fn(padding_items)
        
        # Concatenate the original batch with the new padding data.
        fixed_padded_batch = DataProto.concat([original_batch, padding_data])
        
        print(f"✅ Fix applied. Batch resized from {original_size} to {len(fixed_padded_batch)}.")
        
        # Return the fixed batch and the correct new pad_size.
        return fixed_padded_batch, new_pad_size
    
    # If the original padding was successful, just return the original values.
    return padded_batch, original_pad_size


def _generate_with_gpu_padding(actor_rollout_wg, active_batch: DataProto) -> DataProto:
    """
        Wrapper for generation that handles multi-GPU padding requirements.
        if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
        if active_batch size is not divisible by num_gpus, pad with first sequence
        then remove padding from output
    """
    num_gpus = actor_rollout_wg.config.num_gpus
    if num_gpus <= 1:
        return actor_rollout_wg.generate_sequences(active_batch)
        
    batch_size = active_batch.batch['input_ids'].shape[0]
    remainder = batch_size % num_gpus
    
    for key in active_batch.batch.keys():
        active_batch.batch[key] = active_batch.batch[key].long()
    if remainder == 0:
        return actor_rollout_wg.generate_sequences(active_batch)
    
    # Add padding sequences
    padding_size = num_gpus - remainder
    padded_batch = {}
    
    for k, v in active_batch.batch.items():
        # Use first sequence as padding template
        pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
        padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

    padded_active_batch = DataProto.from_dict(padded_batch)
    for key in padded_active_batch.batch.keys():
        padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

    # Generate with padded batch
    padded_output = actor_rollout_wg.generate_sequences(padded_active_batch)

    # Remove padding from output
    trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
    
    # Handle meta_info if present
    if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
        trimmed_meta = {}
        for k, v in padded_output.meta_info.items():
            if isinstance(v, torch.Tensor):
                trimmed_meta[k] = v[:-padding_size]
            else:
                trimmed_meta[k] = v
        padded_output.meta_info = trimmed_meta
        
    padded_output.batch = trimmed_batch
    return padded_output
