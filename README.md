# HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation

**HiPRAG** (Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation) is a reinforcement learning method designed for training **reasoning-and-searching interleaved LLMs** with improved efficiency and reduced oversearching as well as undersearching behavior. Built upon the foundation of [Search-R1](https://github.com/PeterGriffinJin/Search-R1), HiPRAG introduces hierarchical process reward mechanisms to optimize search strategies and enhance reasoning capabilities.

HiPRAG extends the ideas of **Search-R1** by incorporating intelligent search planning and provides a open-source RL training pipeline for developing more efficient search agent systems.

Paper: [HiPRAG](https://arxiv.org/abs/2510.07794); Code: [GitHub](https://github.com/qualidea1217/HiPRAG); Models: [Hugging Face](https://huggingface.co/collections/qualidea1217/hiprag-68e68c99b8db4d575986c555).

## Key Features
- **Multi-level reasoning**: Implements hierarchical planning to break down complex queries into manageable sub-problems
- **Search optimization**: Reduces oversearching and undersearching through intelligent search planning and validation
- **Step-by-step reasoning**: Maintains structured reasoning with `<think>`, `<step>`, `<reasoning>`, `<search>`, `<context>`, and `<conclusion>` tags

## Installation

We use the exact same environment as [Search-R1](https://github.com/PeterGriffinJin/Search-R1), you can refer to their repository about setting up trainer and retriever environment.

## Quick Start

Train a reasoning + search LLM with HiPRAG's hierarchical process rewards on NQ + HotpotQA dataset with e5 as the retriever and wikipedia as the corpus.

(1) Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Download the NQ + HotpotQA dataset from [Huggingface](https://huggingface.co/datasets/qualidea1217/HiPRAG-Dataset) .

(3) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(4) Fill in the openai api key at the top of the verl/utils/reward_score/qa_em_format.py

(5) Run RL training (PPO) with HiPRAG's reward system.
```bash
conda activate searchr1
bash scripts/nq_hotpotqa/v0.3/train_ppo_format.sh
```

## Inference

#### You can play with the trained HiPRAG model with your own question.

(1) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) Use the function inside inference.py (such as inference_hf_single() for inference single question using huggingface backend), you may need to adjust the parameters in it to fit your environment. For inference on multiple samples, you can also look at the test_template.json for how to organize the data.
```bash
python inference.py
```

## Evaluation

(1) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) Use the function inside analysis.py (such as over_under_search_eval_hf_single() for over and under-search analysis on trajectory of single question using huggingface backend and openai api), you may need to adjust the parameters in it to fit your environment such as openai api key.
```bash
python analysis.py
```

## Acknowledgments

HiPRAG builds upon the excellent work of several open-source projects:

- **Search-R1**: The foundational framework for reasoning-and-searching interleaved LLMs
- **DeepSeek-R1**: Inspiration for reasoning capabilities
- **veRL**: The underlying RL training infrastructure
- **RAGEN**: Components for retrieval-augmented generation

We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.

## Citations

```bibtex
@misc{wu2025hipraghierarchicalprocessrewards,
      title={HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation}, 
      author={Peilin Wu and Mian Zhang and Kun Wan and Wentian Zhao and Kaiyu He and Xinya Du and Zhiyu Chen},
      year={2025},
      eprint={2510.07794},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07794}, 
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and discussions about HiPRAG, please open an issue on GitHub or contact the authors.
