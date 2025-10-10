import httpx
from pandas.core.indexes.base import str_t
import requests
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import torch
import transformers
import asyncio
from transformers.generation.stopping_criteria import StoppingCriteriaList, StopStringCriteria
from typing import Optional
import os
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_async

from prompt import AGENT_PROMPT_V2_SHORT
from utils import extract_str_between, load_jsonl, write_jsonl


_HTTPX_CLIENT_ASYNC: Optional[httpx.AsyncClient] = None


# Helper functions
def extract_search_query(text):
    """Extract the last search query from the text."""
    matches = extract_str_between(text, "<search>", "</search>")
    return matches[-1].strip() if matches else None


def extract_final_answer(text):
    """Extract the final answer from the complete response."""
    matches = extract_str_between(text, "<answer>", "</answer>")
    return matches[-1].strip() if matches else None


def search(query):
    """Perform search using the search endpoint."""
    try:
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        response = requests.post("http://127.0.0.1:8000/retrieve", json=payload)
        results = response.json()['result']
        
        # Format search results
        formatted_results = []
        for idx, doc_item in enumerate(results[0]):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            formatted_results.append(f"Doc {idx+1}(Title: {title}) {text}")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"


async def get_httpx_client_async() -> httpx.AsyncClient:
    """Return a shared AsyncClient, creating it on first use."""
    global _HTTPX_CLIENT_ASYNC
    if _HTTPX_CLIENT_ASYNC is None:
        _HTTPX_CLIENT_ASYNC = httpx.AsyncClient()
    return _HTTPX_CLIENT_ASYNC


async def close_httpx_client_async() -> None:
    """Close and clear the shared AsyncClient if it exists."""
    global _HTTPX_CLIENT_ASYNC
    if _HTTPX_CLIENT_ASYNC is not None:
        await _HTTPX_CLIENT_ASYNC.aclose()
        _HTTPX_CLIENT_ASYNC = None


async def search_async(query: str) -> str:
    """Perform search asynchronously using `httpx.AsyncClient`.

    If a `client` is provided, it will be reused. Otherwise a module-level
    shared client is created on first use and reused across calls.
    """
    try:
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True,
        }
        search_client = await get_httpx_client_async()
        response = await search_client.post("http://127.0.0.1:8000/retrieve", json=payload)

        response.raise_for_status()
        results = response.json()['result']

        formatted_results = []
        for idx, doc_item in enumerate(results[0]):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            formatted_results.append(f"Doc {idx+1}(Title: {title}) {text}")

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"


def inference_hf_single(question: str, model, tokenizer, prompt: str = AGENT_PROMPT_V2_SHORT) -> str:
    """
    Performs inference using an agentic RAG LLM with the specified XML format.
    
    Args:
        question: The user's question to answer
        model_id: The Hugging Face model ID to use
        search_endpoint: The search API endpoint URL
    
    Returns:
        The final answer extracted from the model's response
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Format question
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    
    # Model-specific configurations (adjust based on your model)
    curr_eos = [151645, 151643]  # For Qwen2.5 series models

    # Prepare the initial input (prompt and question only)
    input_text = f"{prompt}\nUser Question: {question}"
    think_step_reasoning = "\n<think>\n<step>\n    <reasoning>"

    # Stop string lists for different tags
    stop_on_search_list = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stop_on_answer_list = ["</answer>", " </answer>", "</answer>\n", " </answer>\n"]

    # Combine all stop strings for the main stopping criteria
    all_stop_strings = stop_on_search_list + stop_on_answer_list
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, all_stop_strings)])
    
    # Apply chat template if available
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        full_response = prompt + think_step_reasoning
    else:
        full_response = input_text + think_step_reasoning
    
    # Main inference loop
    step_count = 0
    max_steps = 10  # Prevent infinite loops
    
    while step_count < max_steps:
        # Encode current prompt - use tokenizer() method for cleaner encoding
        inputs = tokenizer(full_response, return_tensors='pt').to(device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Decode generated tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_response += output_text
        
        # Check if generation ended with </answer> or EOS token (complete response)
        if "</answer>" in output_text or outputs[0][-1].item() in curr_eos:
            break
        
        # Check if we need to perform a search
        search_query = extract_search_query(output_text)
        
        if search_query:  # Continue the generation with search results in context
            search_results = search(search_query)
            full_response += f"\n    <context>{search_results}</context>\n    <conclusion>"
        else:
            full_response += "\n    <conclusion>"
            
        step_count += 1
    
    print(f"Full response: {full_response}")
    return full_response


def inference_hf(question: list[str] | str, model_id: str, tokenizer_id: Optional[str] = None, prompt: str = AGENT_PROMPT_V2_SHORT) -> list[str]:
    """
    Performs inference using an agentic RAG LLM with the specified XML format using Hugging Face Transformers.
    
    Args:
        question: The user's question to answer
        model_id: The Hugging Face model ID to use
        search_endpoint: The search API endpoint URL
    
    Returns:
        The final answer extracted from the model's response
    """
    if isinstance(question, str):
        question = [question]
    
    # Initialize tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id if tokenizer_id else model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    output = []
    for q in tqdm(question):
        output.append(inference_hf_single(q, model, tokenizer, prompt))
    return output


def inference_vllm_single(
    question: str,
    vllm_client: OpenAI,
    model_id: str,
    prompt=AGENT_PROMPT_V2_SHORT,
    tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
):
    """
    Performs inference using an agentic RAG LLM with the specified XML format using vllm server with OpenAI API client.
    
    Args:
        question: The user's question to answer
        client: The OpenAI API client
        prompt: The prompt to use for the inference
        
    Returns:
        The final answer extracted from the model's response
    """
    # Format question
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Stop string lists for different tags (match HF version)
    stop_on_search_list = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stop_on_answer_list = ["</answer>", " </answer>", "</answer>\n", " </answer>\n"]
    all_stop_strings = stop_on_search_list + stop_on_answer_list

    # Prepare the initial input (prompt and question only) and start think/step/reasoning
    input_text = f"{prompt}\nUser Question: {question}"
    think_step_reasoning = "\n<think>\n<step>\n    <reasoning>"
    # Apply chat template if a tokenizer is provided (to mirror HF tokenization formatting)
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        full_response = prompt + think_step_reasoning
    else:
        full_response = input_text + think_step_reasoning

    # Main inference loop
    step_count = 0
    max_steps = 10

    while step_count < max_steps:
        # Generate continuation from the current prefix using Completions API
        completion = vllm_client.completions.create(
            model=model_id,
            prompt=full_response,
            max_tokens=1024,
            stop=all_stop_strings,
            extra_body={"include_stop_str_in_output": True},
        )

        output_text = completion.choices[0].text

        # Append generated text to the running response
        full_response += output_text

        # If the model closed </answer> in this chunk, stop
        if "</answer>" in output_text:
            break

        # Check if we need to perform a search (based on just-generated text)
        search_query = extract_search_query(output_text)

        if search_query:  # Continue the generation with search results in context
            search_results = search(search_query)
            full_response += f"\n    <context>{search_results}</context>\n    <conclusion>"
        else:
            full_response += "\n    <conclusion>"

        step_count += 1

    print(f"Full response: {full_response}")
    return full_response


def inference_vllm(question: list[str] | str, api_key: str, base_url: str, model_id: str, tokenizer_id: Optional[str] = None, prompt=AGENT_PROMPT_V2_SHORT):
    """
    Performs inference using an agentic RAG LLM with the specified XML format using vllm server with OpenAI API client.
    
    Args:
        question: The user's question to answer
        api_key: The API key for OpenAI
        base_url: The base URL for OpenAI
        prompt: The prompt to use for the inference
        
    Returns:
        The final answer extracted from the model's response
    """
    vllm_client = OpenAI(api_key=api_key, base_url=base_url)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id if tokenizer_id else model_id)
    if isinstance(question, str):
        question = [question]
    output = []
    for q in tqdm(question):
        output.append(inference_vllm_single(q, vllm_client, model_id, prompt, tokenizer))
    return output
    

# Async variants using AsyncOpenAI and search_async
async def inference_vllm_single_async(question: str, vllm_client: AsyncOpenAI, model_id: str, prompt=AGENT_PROMPT_V2_SHORT, tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None) -> str:
    """
    Async version of `inference_vllm_single` using AsyncOpenAI and `search_async`.
    Mirrors the exact logic of the sync version.
    """
    # Format question
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Stop string lists for different tags (match HF version)
    stop_on_search_list = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stop_on_answer_list = ["</answer>", " </answer>", "</answer>\n", " </answer>\n"]
    all_stop_strings = stop_on_search_list + stop_on_answer_list

    # Prepare the initial input (prompt and question only) and start think/step/reasoning
    input_text = f"{prompt}\nUser Question: {question}"
    think_step_reasoning = "\n<think>\n<step>\n    <reasoning>"
    # Apply chat template if a tokenizer is provided (to mirror HF tokenization formatting)
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        full_response = prompt + think_step_reasoning
    else:
        full_response = input_text + think_step_reasoning

    # Main inference loop
    step_count = 0
    max_steps = 10

    while step_count < max_steps:
        # Generate continuation from the current prefix using Completions API
        completion = await vllm_client.completions.create(
            model=model_id,
            prompt=full_response,
            max_tokens=1024,
            stop=all_stop_strings,
            extra_body={"include_stop_str_in_output": True},
        )

        output_text = completion.choices[0].text

        # Append generated text to the running response
        full_response += output_text

        # If the model closed </answer> in this chunk, stop
        if "</answer>" in output_text:
            break

        # Check if we need to perform a search (based on just-generated text)
        search_query = extract_search_query(output_text)

        if search_query:  # Continue the generation with search results in context
            search_results = await search_async(search_query)
            full_response += f"\n    <context>{search_results}</context>\n    <conclusion>"
        else:
            full_response += "\n    <conclusion>"

        step_count += 1

    # print(f"Full response: {full_response}")
    return full_response


async def inference_vllm_async(question: list[str] | str, api_key: str, base_url: str, model_id: str, tokenizer_id: Optional[str] = None, prompt=AGENT_PROMPT_V2_SHORT, max_concurrency: int = 64) -> list[str]:
    """
    Async version of `inference_vllm` using AsyncOpenAI and `inference_vllm_single_async`.
    Mirrors the exact logic of the sync version.
    Note: if the output is weird, try to disable cascade attention when launching the vllm server.
    """
    vllm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id if tokenizer_id else model_id)
    if isinstance(question, str):
        question = [question]
    # Bounded concurrency using a semaphore to avoid overloading backend services
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(q: str) -> str:
        async with semaphore:
            return await inference_vllm_single_async(q, vllm_client, model_id, prompt, tokenizer)

    tasks = [run_one(q) for q in question]
    results = await tqdm_async.gather(*tasks)
    # Close the async search client
    await close_httpx_client_async()
    return list(results)


def inference():
    BASE_URL = "http://localhost:8001/v1"
    MODEL_ID = "/home/pxw240002/Search-R1/verl_checkpoints/nq_hotpotqa_train-search-r1-grpo-qwen2.5-3b-it-em-structureformat-step-wiser/actor/global_step_200/"
    API_KEY = "EMPTY"
    MAX_CONCURRENCY = 64

    data_list = load_jsonl("results/test_template.jsonl")
    questions = [data["question"] for data in data_list]

    # results = inference_vllm(questions, API_KEY, BASE_URL, MODEL_ID, MODEL_ID, AGENT_PROMPT_V2_SHORT)
    results = asyncio.run(
        inference_vllm_async(
            question=questions,
            api_key=API_KEY,
            base_url=BASE_URL,
            model_id=MODEL_ID,
            tokenizer_id=MODEL_ID,
            prompt=AGENT_PROMPT_V2_SHORT,
            max_concurrency=MAX_CONCURRENCY,
        )
    )

    for data, result in zip(data_list, results):
        data["result"] = result
    write_jsonl(data_list, "results/qwen2.5-3b-it-grpo-wiser-step-200.jsonl")
    print(f"Wrote {len(data_list)} rows with result to: results/qwen2.5-3b-it-grpo-wiser-step-200.jsonl")


# Example usage
if __name__ == "__main__":
    inference()    
