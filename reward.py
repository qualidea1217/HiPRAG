import re
import string


def extract_str_between_optimized(text: str, start: str, end: str) -> list[str]:
    """Extract strings between start and end markers (non-inclusive)."""
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


def normalize_string(str: str) -> str:
    """Normalize answer by removing articles, punctuation, extra spaces, and lowercasing."""
    # Remove articles, punctuation, lowercase, and fix whitespace in one pass
    str = re.sub(r'\b(a|an|the)\b', ' ', str.lower())
    str = ''.join(c if c not in string.punctuation else ' ' for c in str)
    return ' '.join(str.split())


def cover_exact_match(pred: str, gold: str) -> bool:
    """Check if any golden answer is a substring of the prediction."""
    if not pred:
        return False
    if isinstance(gold, str):
        gold = [gold]
    norm_pred = normalize_string(pred)
    return any(normalize_string(ans) in norm_pred for ans in gold)


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
    
    # # Check no extra content after </answer>
    # remaining = answer_section[answer_end + 9:].strip()
    # if remaining:
    #     return False
    
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


def search_r1_format(model_response: str, ground_truth_answer: str, lambda_f: float = 0.2) -> float:
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
    answer_matches = extract_str_between_optimized(model_response, '<answer>', '</answer>')
    predicted_answer = answer_matches[-1].strip() if answer_matches else ""
    
    # Check correctness and format
    answer_correct = cover_exact_match(predicted_answer, ground_truth_answer)
    format_correct = check_full_response_format_optimized(model_response)
    
    # Apply Equation 3
    if answer_correct:
        return 1.0 if format_correct else 1.0 - lambda_f
    else:
        return lambda_f if format_correct else 0.0
    

if __name__ == "__main__":
    def run_test(name, response, expected):
        result = check_full_response_format_optimized(response)
        print(f"{name}: {'PASS' if result == expected else 'FAIL'} (Expected {expected}, Got {result})")

    # Valid: single step (with search/context)
    run_test(
        "Valid single step (with search/context)",
        '<think><step><reasoning>Reasoning here.</reasoning><search>search query</search><context>context here</context><conclusion>conclusion here</conclusion></step></think><answer>Final answer here.</answer>',
        True
    )
    # Valid: single step (no search/context)
    run_test(
        "Valid single step (no search/context)",
        '<think><step><reasoning>Reasoning here.</reasoning><conclusion>conclusion here</conclusion></step></think><answer>Final answer here.</answer>',
        True
    )
    # Valid: multiple steps (mixed)
    run_test(
        "Valid multiple steps (mixed)",
        '<think><step><reasoning>Step 1 reasoning.</reasoning><search>query1</search><context>context1</context><conclusion>conclusion1</conclusion></step><step><reasoning>Step 2 reasoning.</reasoning><conclusion>conclusion2</conclusion></step></think><answer>Multi-step answer.</answer>',
        True
    )
    # Valid: whitespace (no search/context)
    run_test(
        "Valid with whitespace (no search/context)",
        '''\n        <think>\n            <step>\n                <reasoning>  Reasoning.  </reasoning>\n                <conclusion>  Done.  </conclusion>\n            </step>\n        </think>\n        <answer>  Answer.  </answer>\n        ''',
        True
    )
    # Valid: empty search/context (should be invalid in V2)
    run_test(
        "Invalid empty search/context (V2)",
        '<think><step><reasoning>Reasoning</reasoning><search></search><context></context><conclusion>conclusion</conclusion></step></think><answer>Answer.</answer>',
        False
    )

    # Valid: long content
    reasoning = 'A' * 1000
    search = 'B' * 1000
    context = 'C' * 1000
    conclusion = 'D' * 1000
    answer = 'E' * 1000
    run_test(
        "Valid long content",
        f'<think><step><reasoning>{reasoning}</reasoning><search>{search}</search><context>{context}</context><conclusion>{conclusion}</conclusion></step></think><answer>{answer}</answer>',
        True
    )

    # Invalid: missing reasoning
    run_test(
        "Invalid missing reasoning",
        '<think><step><search>query</search><context>context</context><conclusion>conclusion</conclusion></step></think><answer>Answer.</answer>',
        False
    )

    # Invalid: missing step
    run_test(
        "Invalid missing step",
        '<think></think><answer>Answer.</answer>',
        False
    )

    # Invalid: wrong order
    run_test(
        "Invalid wrong order",
        '<think><step><search>query</search><reasoning>Reasoning</reasoning><context>context</context><conclusion>conclusion</conclusion></step></think><answer>Answer.</answer>',
        False
    )

    # Invalid: extra content outside
    run_test(
        "Invalid extra content outside",
        'Extra text<think><step><reasoning>Reasoning</reasoning><search>query</search><context>context</context><conclusion>conclusion</conclusion></step></think><answer>Answer.</answer>',
        True
    )

    # Invalid: no answer
    run_test(
        "Invalid no answer",
        '<think><step><reasoning>Reasoning</reasoning><search>query</search><context>context</context><conclusion>conclusion</conclusion></step></think>',
        False
    )

    # Invalid: no think
    run_test(
        "Invalid no think",
        '<step><reasoning>Reasoning</reasoning><search>query</search><context>context</context><conclusion>conclusion</conclusion></step><answer>Answer.</answer>',
        False
    )

    # Invalid: nested steps
    run_test(
        "Invalid nested steps",
        '<think><step><reasoning>Reasoning</reasoning><step><search>query</search><context>context</context><conclusion>conclusion</conclusion></step></step></think><answer>Answer.</answer>',
        False
    )

    # Invalid: duplicate tags
    run_test(
        "Invalid duplicate tags",
        '<think><step><reasoning>Reasoning</reasoning><search>query</search><search>another</search><context>context</context><conclusion>conclusion</conclusion></step></think><answer>Answer.</answer>',
        False
    )
    
    # Performance test for search_r1_format()
    print("\n" + "="*50)
    print("PERFORMANCE TEST FOR search_r1_format()")
    print("="*50)
    
    import time
    import random
    import string
    
    def generate_test_data(size="medium"):
        """Generate test data of different sizes"""
        if size == "small":
            reasoning = "Short reasoning."
            search = "Short search."
            context = "Short context."
            conclusion = "Short conclusion."
            answer = "Short answer."
        elif size == "medium":
            reasoning = "This is a medium-length reasoning that contains multiple sentences and some complexity to test the performance of the search_r1_format function under normal conditions."
            search = "Medium search query with some complexity and multiple terms that might be used in a real-world scenario."
            context = "Medium context with relevant information that would typically be retrieved from a search operation."
            conclusion = "Medium conclusion that summarizes the findings and provides a clear answer based on the reasoning and context."
            answer = "Medium answer that directly addresses the question asked."
        else:  # large
            reasoning = "A" * 1000
            search = "B" * 1000
            context = "C" * 1000
            conclusion = "D" * 1000
            answer = "E" * 1000
        
        return f'<think><step><reasoning>{reasoning}</reasoning><search>{search}</search><context>{context}</context><conclusion>{conclusion}</conclusion></step></think><answer>{answer}</answer>'
    
    def performance_test():
        """Run performance tests for search_r1_format()"""
        test_cases = [
            ("Small input", generate_test_data("small"), "Test answer"),
            ("Medium input", generate_test_data("medium"), "Test answer"),
            ("Large input", generate_test_data("large"), "Test answer"),
        ]
        
        # Warm up
        print("Warming up...")
        for _ in range(100):
            search_r1_format(generate_test_data("medium"), "Test answer")
        
        print("\nRunning performance tests...")
        results = {}
        
        for name, response, gt_answer in test_cases:
            print(f"\nTesting {name}:")
            
            # Test multiple runs for accuracy
            times = []
            for i in range(1000):
                start_time = time.perf_counter()
                result = search_r1_format(response, gt_answer)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"  Average time: {avg_time*1000:.4f} ms")
            print(f"  Min time: {min_time*1000:.4f} ms")
            print(f"  Max time: {max_time*1000:.4f} ms")
            print(f"  Function result: {result}")
            
            results[name] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'result': result
            }
        
        # Summary
        print("\n" + "-"*50)
        print("PERFORMANCE SUMMARY")
        print("-"*50)
        for name, data in results.items():
            print(f"{name}: {data['avg_time']*1000:.4f} ms avg")
        
        # Throughput test
        print("\n" + "-"*50)
        print("THROUGHPUT TEST")
        print("-"*50)
        
        test_response = generate_test_data("medium")
        test_answer = "Test answer"
        
        start_time = time.perf_counter()
        iterations = 100000
        for _ in range(iterations):
            search_r1_format(test_response, test_answer)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = iterations / total_time
        
        print(f"Processed {iterations:,} calls in {total_time:.4f} seconds")
        print(f"Throughput: {throughput:.0f} calls/second")
        print(f"Average time per call: {(total_time/iterations)*1000:.4f} ms")
    
    # Run the performance test
    performance_test()