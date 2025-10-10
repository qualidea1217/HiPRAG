import asyncio
from typing import Optional
import os
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
import torch
import transformers
from transformers.generation.stopping_criteria import StoppingCriteriaList, StopStringCriteria
from tqdm.asyncio import tqdm as tqdm_async

from prompt import SEARCH_STEP_VERIFY_PROMPT_V1, NON_SEARCH_STEP_VERIFY_PROMPT_V1, AGENT_PROMPT_V2_SHORT
from reward import check_full_response_format, cover_exact_match, check_full_response_format_optimized
from utils import extract_str_between, load_jsonl, write_jsonl


gpt_client = OpenAI(api_key="")
gpt_client_async = AsyncOpenAI(api_key="")
gpt_client_azure = AzureOpenAI(
    azure_endpoint="",
    api_key="",
    api_version=""
)
gpt_client_azure_async = AsyncAzureOpenAI(
    azure_endpoint="",
    api_key="",
    api_version=""
)


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


def test_oversearch_direct_hf(step_content: str, model, tokenizer, gpt_client: OpenAI) -> bool:
    """
    Test the oversearch behavior of a single step by directly asking the search query to the model.
    """
    # Extract the search query and original conclusion of this step
    try:
        search_query = extract_str_between(step_content, "<search>", "</search>")[-1]
        conclusion_content = extract_str_between(step_content, "<conclusion>", "</conclusion>")[-1]
    except Exception as e:
        print(f"Error: {e}")
        return False
    # Apply chat template if available
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": f"Please answer the following question or provide relevant information to the statement: {search_query}"}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt = search_query
    # Generate response by directly asking the search query to the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # Determine if the two conclusions are equivalent
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStatement 1: {conclusion_content}\n\nStatement 2: {output_text}"}
            ]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True if "True" in is_correct else False


def test_oversearch_direct_vllm(step_content: str, vllm_client: OpenAI, gpt_client: OpenAI, model_id: str) -> bool:
    """
    Test the oversearch behavior of a single step by directly asking the search query to the model.
    """
    # Extract the search query and original conclusion of this step
    try:
        search_query = extract_str_between(step_content, "<search>", "</search>")[-1]
        conclusion_content = extract_str_between(step_content, "<conclusion>", "</conclusion>")[-1]
    except Exception as e:
        print(f"Error: {e}")
        return False
    # Generate response by directly asking the search query to the model
    completion = vllm_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": f"Please answer the following question or provide relevant information to the statement: {search_query}"}],
        max_tokens=1024,
    )
    output_text = completion.choices[0].message.content
    # Determine if the two conclusions are equivalent
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStatement 1: {conclusion_content}\n\nStatement 2: {output_text}"}]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True if "True" in is_correct else False


async def test_oversearch_direct_vllm_async(step_content: str, vllm_client: AsyncOpenAI, gpt_client: AsyncOpenAI, model_id: str) -> bool:
    """
    Test the oversearch behavior of a single step by directly asking the search query to the model.
    """
    # Extract the search query and original conclusion of this step
    try:
        search_query = extract_str_between(step_content, "<search>", "</search>")[-1]
        conclusion_content = extract_str_between(step_content, "<conclusion>", "</conclusion>")[-1]
    except Exception as e:
        print(f"Error: {e}")
        return False
    # Generate response by directly asking the search query to the model
    completion = await vllm_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": f"Please answer the following question or provide relevant information to the statement: {search_query}"}],
        max_tokens=1024,
    )
    output_text = completion.choices[0].message.content
    # Determine if the two conclusions are equivalent
    try:
        completion = await gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStatement 1: {conclusion_content}\n\nStatement 2: {output_text}"}]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True if "True" in is_correct else False


def test_oversearch_override_hf(step_content: str, step_start: int, full_response: str, model, tokenizer, gpt_client: OpenAI) -> bool:
    """
    Test the oversearch behavior of a single step by overriding the searching behavior and generating the conclusion.
    """
    # Find the reasoning end
    reasoning_end = step_content.find('</reasoning>')
    # Keep only up to </reasoning> and add <conclusion> tag
    modified_step_content = step_content[:reasoning_end + 12] + "\n    <conclusion>"
    # Create the modified full response - stop at conclusion, don't include anything after
    modified_response = full_response[:step_start + 6] + modified_step_content
    # Set up stopping criteria for regeneration
    stop_strings = ["</conclusion>", " </conclusion>", "</conclusion>\n", " </conclusion>\n", "</step>", " </step>", "</step>\n", " </step>\n"]
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, stop_strings)])
    # Generate response with forced not to search
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(modified_response, return_tensors='pt').to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # Update the modified response with the regenerated content
    modified_response = modified_response + output_text
    # Extract the content in <conclusion> tags from the newly generated text
    not_search_conclusion_end = output_text.find("</conclusion>")
    not_search_conclusion_content = output_text[:not_search_conclusion_end] if not_search_conclusion_end != -1 else output_text
    conclusion_content = extract_str_between(step_content, "<conclusion>", "</conclusion>")[-1]
    # Determine if the two conclusions are equivalent
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStatement 1: {conclusion_content}\n\nStatement 2: {not_search_conclusion_content}"}
            ]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
        return True if "True" in is_correct else False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_oversearch_override_vllm(step_content: str, step_start: int, full_response: str, vllm_client: OpenAI, gpt_client: OpenAI, model_id: str) -> bool:
    """
    Test the oversearch behavior of a single step by overriding the searching behavior and generating the conclusion.
    """
    # Find the reasoning end
    reasoning_end = step_content.find('</reasoning>')
    # Keep only up to </reasoning> and add <conclusion> tag
    modified_step_content = step_content[:reasoning_end + 12] + "\n    <conclusion>"
    # Create the modified full response - stop at conclusion, don't include anything after
    modified_response = full_response[:step_start + 6] + modified_step_content
    # Set up stopping criteria for regeneration
    stop_strings = ["</conclusion>", " </conclusion>", "</conclusion>\n", " </conclusion>\n", "</step>", " </step>", "</step>\n", " </step>\n"]
    # Generate response with forced not to search
    output_text = vllm_client.completions.create(
        model=model_id,
        prompt=modified_response,
        max_tokens=1024,
        stop=stop_strings,
        extra_body={"include_stop_str_in_output": True},
    )
    output_text = output_text.choices[0].text
    # Update the modified response with the regenerated content
    modified_response = modified_response + output_text
    # Extract the content in <conclusion> tags from the newly generated text
    not_search_conclusion_end = output_text.find("</conclusion>")
    not_search_conclusion_content = output_text[:not_search_conclusion_end] if not_search_conclusion_end != -1 else output_text
    conclusion_content = extract_str_between(step_content, "<conclusion>", "</conclusion>")[-1]
    # Determine if the two conclusions are equivalent
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStatement 1: {conclusion_content}\n\nStatement 2: {not_search_conclusion_content}"}]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
        return True if "True" in is_correct else False
    except Exception as e:
        print(f"Error: {e}")
        return False


async def test_oversearch_override_vllm_async(step_content: str, step_start: int, full_response: str, vllm_client: AsyncOpenAI, gpt_client: AsyncOpenAI, model_id: str) -> bool:
    """
    Test the oversearch behavior of a single step by overriding the searching behavior and generating the conclusion.
    """
    # Find the reasoning end
    reasoning_end = step_content.find('</reasoning>')
    # Keep only up to </reasoning> and add <conclusion> tag
    modified_step_content = step_content[:reasoning_end + 12] + "\n    <conclusion>"
    # Create the modified full response - stop at conclusion, don't include anything after
    modified_response = full_response[:step_start + 6] + modified_step_content
    # Set up stopping criteria for regeneration
    stop_strings = ["</conclusion>", " </conclusion>", "</conclusion>\n", " </conclusion>\n", "</step>", " </step>", "</step>\n", " </step>\n"]
    # Generate response with forced not to search
    output_text = await vllm_client.completions.create(
        model=model_id,
        prompt=modified_response,
        max_tokens=1024,
        stop=stop_strings,
        extra_body={"include_stop_str_in_output": True},
    )
    output_text = output_text.choices[0].text
    # Update the modified response with the regenerated content
    modified_response = modified_response + output_text
    # Extract the content in <conclusion> tags from the newly generated text
    not_search_conclusion_end = output_text.find("</conclusion>")
    not_search_conclusion_content = output_text[:not_search_conclusion_end] if not_search_conclusion_end != -1 else output_text
    conclusion_content = extract_str_between(step_content, "<conclusion>", "</conclusion>")[-1]
    # Determine if the two conclusions are equivalent
    try:
        completion = await gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStatement 1: {conclusion_content}\n\nStatement 2: {not_search_conclusion_content}"}]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
        return True if "True" in is_correct else False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_undersearch(step_content: str, gpt_client: OpenAI) -> bool:
    """
    Test the undersearch behavior of a single step using openai client.
    """
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": NON_SEARCH_STEP_VERIFY_PROMPT_V1 + f"\n\nStep Content: {step_content}"}
            ]
        )
        result = completion.choices[0].message.content
        is_correct = extract_str_between(result, "<answer>", "</answer>")[-1]
        return False if "True" in is_correct else True
    except Exception as e:
        print(f"Error: {e}")
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
        print(f"Error: {e}")
        return False


def over_under_search_eval_hf_single(full_response: str, model, tokenizer, gpt_client: OpenAI) -> list[str]:
    """
    Evaluate the oversearch behavior of a single trajectory using Hugging Face Transformers and openai client.

    Args:
        question: The user's question to answer
        model: The Hugging Face model to use
        tokenizer: The Hugging Face tokenizer to use
        prompt: The prompt to use for the model
    """
    if not check_full_response_format(full_response):
        return "Invalid response format, unable to evaluate"

    step_eval_results = []
    step_content_list, step_start_list = get_steps_from_full_response(full_response)
    for step_content, step_start in zip(step_content_list, step_start_list):
        if "<search>" in step_content:  # search step
            is_oversearched = test_oversearch_direct_hf(step_content, model, tokenizer, gpt_client)
            step_eval_results.append("oversearch" if is_oversearched else "search")
            print(f"Search step is {'over' if is_oversearched else 'not over'}-searched")
        else:  # non-search step
            is_undersearched = test_undersearch(step_content, gpt_client)
            step_eval_results.append("undersearch" if is_undersearched else "non-search")
            print(f"Non-search step is {'under' if is_undersearched else 'not under'}-searched")
    return step_eval_results


def over_under_search_eval_vllm_single(full_response: str, vllm_client: OpenAI, gpt_client: OpenAI, model_id: str) -> list[str]:
    """
    Evaluate the oversearch behavior of a single trajectory using VLLM and openai client.
    """
    if not check_full_response_format(full_response):
        return "Invalid response format, unable to evaluate"

    step_eval_results = []
    step_content_list, step_start_list = get_steps_from_full_response(full_response)
    for step_content, step_start in zip(step_content_list, step_start_list):
        if "<search>" in step_content:  # search step
            is_oversearched = test_oversearch_direct_vllm(step_content, vllm_client, gpt_client, model_id)
            step_eval_results.append("oversearch" if is_oversearched else "search")
            print(f"Search step is {'over' if is_oversearched else 'not over'}-searched")
        else:  # non-search step
            is_undersearched = test_undersearch_async(step_content, gpt_client)
            step_eval_results.append("undersearch" if is_undersearched else "non-search")
            print(f"Non-search step is {'under' if is_undersearched else 'not under'}-searched")
    return step_eval_results


async def over_under_search_eval_vllm_single_async(full_response: str, vllm_client: AsyncOpenAI, gpt_client: AsyncOpenAI, model_id: str) -> list[str]:
    """
    Evaluate the oversearch behavior of a single trajectory using async VLLM and async openai client.
    """
    if not check_full_response_format(full_response):
        return "Invalid response format, unable to evaluate"

    step_eval_results = []
    step_content_list, step_start_list = get_steps_from_full_response(full_response)
    for step_content, step_start in zip(step_content_list, step_start_list):
        if "<search>" in step_content:  # search step
            is_oversearched = await test_oversearch_direct_vllm_async(step_content, vllm_client, gpt_client, model_id)
            step_eval_results.append("oversearch" if is_oversearched else "search")
            # print(f"Search step is {'over' if is_oversearched else 'not over'}-searched")
        else:  # non-search step
            is_undersearched = await test_undersearch_async(step_content, gpt_client)
            step_eval_results.append("undersearch" if is_undersearched else "non-search")
            # print(f"Non-search step is {'under' if is_undersearched else 'not under'}-searched")
    return step_eval_results


def over_under_search_eval_hf(full_responses: list[str] | str, model_id: str, gpt_client: OpenAI, tokenizer_id: Optional[str] = None) -> list[list[str]]:
    """
    Evaluate the oversearch behavior of a list of trajectories using Hugging Face Transformers and openai client.
    """
    if isinstance(full_responses, str):
        full_responses = [full_responses]

    if tokenizer_id is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)

    trajectory_eval_results = []
    for full_response in full_responses:
        trajectory_eval_results.append(over_under_search_eval_hf_single(full_response, model, tokenizer, gpt_client))
    return trajectory_eval_results


def over_under_search_eval_vllm(full_responses: list[str] | str, vllm_client: OpenAI, gpt_client: OpenAI, model_id: str) -> list[list[str]]:
    """
    Evaluate the oversearch behavior of a list of trajectories using VLLM and openai client.
    """
    if isinstance(full_responses, str):
        full_responses = [full_responses]

    trajectory_eval_results = []
    for full_response in full_responses:
        trajectory_eval_results.append(over_under_search_eval_vllm_single(full_response, vllm_client, gpt_client, model_id))
    return trajectory_eval_results


async def over_under_search_eval_vllm_async(full_responses: list[str] | str, vllm_client: AsyncOpenAI, gpt_client: AsyncOpenAI, model_id: str, max_concurrency: int = 64) -> list[list[str]]:
    """
    Evaluate the oversearch behavior of a list of trajectories using async VLLM and async openai client.
    """
    if isinstance(full_responses, str):
        full_responses = [full_responses]

    semaphore = asyncio.Semaphore(max_concurrency)
    async def eval_single_trajectory(full_response: str):
        async with semaphore:
            return await over_under_search_eval_vllm_single_async(full_response, vllm_client, gpt_client, model_id)

    tasks = [eval_single_trajectory(full_response) for full_response in full_responses]
    results = await tqdm_async.gather(*tasks)
    return list(results)


def over_under_search_eval():
    BASE_URL = "http://localhost:8001/v1"
    MODEL_ID = "/home/pxw240002/models/qualidea1217/nq_hotpotqa_train-search-r1-ppo-llama3.2-3b-it-em-structureformat-step/actor/global_step_200/"
    API_KEY = "EMPTY"
    MAX_CONCURRENCY = 64

    vllm_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    data_list = load_jsonl("llama3.2-3b-it-ppo-baseline-step-200.jsonl")
    full_responses = [data["result"] for data in data_list]

    # results = inference_vllm(questions, API_KEY, BASE_URL, MODEL_ID, MODEL_ID, AGENT_PROMPT_V2_SHORT)
    results = asyncio.run(
        over_under_search_eval_vllm_async(
            full_responses=full_responses,
            vllm_client=vllm_client,
            gpt_client=gpt_client_azure_async,
            model_id=MODEL_ID,
            max_concurrency=MAX_CONCURRENCY,
        )
    )

    for data, result in zip(data_list, results):
        data["step_eval_results"] = result
    write_jsonl(data_list, "llama3.2-3b-it-ppo-baseline-step-200.jsonl")
    print(f"Wrote {len(data_list)} rows with result to: llama3.2-3b-it-ppo-baseline-step-200.jsonl")


async def inference_then_eval_vllm_async(question: list[str] | str, api_key: str, base_url: str, model_id: str, tokenizer_id: Optional[str] = None, prompt=AGENT_PROMPT_V2_SHORT, max_concurrency: int = 64):
    from inference import inference_vllm_single_async, close_httpx_client_async

    vllm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id if tokenizer_id else model_id)
    if isinstance(question, str):
        question = [question]

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(q: str) -> tuple[str, list[str]]:
        async with semaphore:
            trajectory = await inference_vllm_single_async(q, vllm_client, model_id, prompt, tokenizer)
            step_eval_results = await over_under_search_eval_vllm_single_async(trajectory, vllm_client, gpt_client_async, model_id)
            return (trajectory, step_eval_results)

    tasks = [run_one(q) for q in question]
    results = await tqdm_async.gather(*tasks)
    await close_httpx_client_async()
    return list(results)


def inference_then_eval():
    BASE_URL = "http://localhost:8001/v1"
    MODEL_ID = "/home/pxw240002/models/qualidea1217/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-em-structureformat-step-wiser-threshold-0.2/global_step_150/"
    API_KEY = "EMPTY"
    MAX_CONCURRENCY = 64

    data_list = load_jsonl("results/test_template.jsonl")
    questions = [data["question"] for data in data_list]

    results = asyncio.run(
        inference_then_eval_vllm_async(
            question=questions,
            api_key=API_KEY,
            base_url=BASE_URL,
            model_id=MODEL_ID,
            max_concurrency=MAX_CONCURRENCY,
        )
    )

    for data, result in zip(data_list, results):
        data["result"] = result[0]
        data["step_eval_results"] = result[1]
    write_jsonl(data_list, "results/qwen2.5-3b-it-ppo-wiser-threshold-0.2-step-150.jsonl")
    print(f"Wrote {len(data_list)} rows with result to: results/qwen2.5-3b-it-ppo-wiser-threshold-0.2-step-150.jsonl")


def get_scores(result_file_path: str):
    data_list = load_jsonl(result_file_path)
    print(f"Loaded {len(data_list)} rows from: {result_file_path}")
    data_source_list = sorted(list(set([data["data_source"] for data in data_list])))
    total_correct_count = 0
    total_search_count, total_non_search_count, total_over_search_count, total_under_search_count = 0, 0, 0, 0
    for data_source in data_source_list:
        data_list_source = [data for data in data_list if data["data_source"] == data_source]
        correct_count = 0
        search_count, non_search_count, over_search_count, under_search_count = 0, 0, 0, 0
        for data in data_list_source:
            pred_answer = extract_str_between(data["result"], "<answer>", "</answer>")
            if pred_answer:
                is_correct = cover_exact_match(pred_answer[-1], data["golden_answers"])
                if is_correct:
                    correct_count += 1
            step_eval_results = data.get("step_eval_results", [])
            if step_eval_results:
                for step_eval_result in step_eval_results:
                    if step_eval_result == "search":
                        search_count += 1
                    elif step_eval_result == "non-search":
                        non_search_count += 1
                    elif step_eval_result == "oversearch":
                        over_search_count += 1
                    elif step_eval_result == "undersearch":
                        under_search_count += 1
        over_search_rate = over_search_count / (search_count + over_search_count) if search_count + over_search_count > 0 else 0
        under_search_rate = under_search_count / (non_search_count + under_search_count) if non_search_count + under_search_count > 0 else 0
        print(f"{data_source}: {round(correct_count / len(data_list_source), 3)}, over_search_rate: {round(over_search_rate, 3)}, under_search_rate: {round(under_search_rate, 3)}")
        total_correct_count += correct_count
        total_search_count += search_count
        total_non_search_count += non_search_count
        total_over_search_count += over_search_count
        total_under_search_count += under_search_count
    print(f"Total: {round(total_correct_count / len(data_list), 3)}, over_search_rate: {round(total_over_search_count / (total_search_count + total_over_search_count) if total_search_count + total_over_search_count > 0 else 0, 3)}, under_search_rate: {round(total_under_search_count / (total_non_search_count + total_under_search_count) if total_non_search_count + total_under_search_count > 0 else 0, 3)}")


if __name__ == "__main__":
    over_under_search_eval()
    # inference_then_eval()
    # get_scores("results/qwen2.5-3b-it-ppo-wiser-step-400.jsonl")
