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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import asyncio
import string
import random
import torch
import logging
from typing import Union, List
from tensordict import TensorDict

logger = logging.getLogger(__name__)
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem, collate_fn
import torch
from verl.utils.reward_score import qa_em, qa_em_format
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
        return qa_em_format.compute_score_em
    else:
        raise NotImplementedError


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
        # gen_output = _generate_with_gpu_padding(actor_rollout_wg, gen_batch)
        
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
    print(f"world_size: {actor_rollout_wg.world_size}")
    num_gpus = actor_rollout_wg.world_size
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


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score

    async def _evaluate_batch_with_semaphore(self, tasks, max_concurrency=64):
        """Evaluate tasks with controlled concurrency using semaphore."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_task(task) for task in tasks]
        return await asyncio.gather(*bounded_tasks)

    def __call__(self, data: DataProto, actor_rollout_wg=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}

        # for over-search, batch generate all the search queries and save the result for future reward computation
        all_search_query_list = []
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
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

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

            think_list = extract_str_between(sequences_str, "<think>", "</think>")
            if think_list:
                search_query_list = extract_str_between(think_list[-1], "<search>", "</search>")
                if search_query_list:
                    all_search_query_list.extend(search_query_list)

        all_search_query_list = list(set(all_search_query_list))
        all_search_query_list_chat_templated = [self.tokenizer.apply_chat_template([{"role": "user", "content": f"Please answer the following question or provide relevant information to the statement: {search_query}"}], add_generation_prompt=True, tokenize=False) for search_query in all_search_query_list]

        outputs = []
        if actor_rollout_wg and self.tokenizer:
            input_prompts, outputs = generate_extra_outputs_for_qa_em(
                input_strings=all_search_query_list_chat_templated,
                actor_rollout_wg=actor_rollout_wg,
                tokenizer=self.tokenizer,
                temperature=1.0,
                max_prompt_length=512
            )

        if outputs:
            # for prompt, generated_text in zip(all_search_query_list, outputs):
            #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
            all_search_query_dict = {prompt: generated_text for prompt, generated_text in zip(all_search_query_list, outputs)}
            print("search query dict created successfully")
        else:
            all_search_query_dict = {}
            print("search query dict created failed")

        # evaluate the over-search and under-search and calculate the reward asynchronously
        tasks = []
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
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            tasks.append(compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
                                        structure_format_score=self.structure_format_score, 
                                        final_format_score=self.final_format_score, 
                                        retrieval_score=self.retrieval_score,
                                        format_score=self.format_score,
                                        actor_rollout_wg=actor_rollout_wg,
                                        tokenizer=self.tokenizer,
                                        all_search_query_dict=all_search_query_dict))

        # Use semaphore to control max concurrency
        reward_score_list = asyncio.run(self._evaluate_batch_with_semaphore(tasks, max_concurrency=128))

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            # valid_response_ids = response_ids[:valid_response_length]

            # # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # sequences_str = self.tokenizer.decode(sequences)

            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # # select rm_score
            # data_source = data_item.non_tensor_batch['data_source']
            # compute_score_fn = _select_rm_score_fn(data_source)

            # score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
            #                          structure_format_score=self.structure_format_score, 
            #                          final_format_score=self.final_format_score, 
            #                          retrieval_score=self.retrieval_score,
            #                          format_score=self.format_score,
            #                          actor_rollout_wg=actor_rollout_wg,
            #                          tokenizer=self.tokenizer,
            #                          all_search_query_dict=all_search_query_dict)

            reward_tensor[i, valid_response_length - 1] = reward_score_list[i]
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              structure_format_score=config.reward_model.structure_format_score, 
                              final_format_score=config.reward_model.final_format_score,
                              retrieval_score=config.reward_model.retrieval_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
