import torch
import random 
import re
from tqdm.notebook import tqdm

def random_split_dataset(lst, test_samples):
    """
    Splits a list into two parts based on given number of samples wanted for test
    """
    # Create a copy and shuffle it
    shuffled_list = lst[:]
    random.shuffle(shuffled_list)
    
    # Slice the list
    test_set = shuffled_list[:test_samples]
    train_set = shuffled_list[test_samples:]
    
    return train_set, test_set

def extract_answer(text):
    """
    Extract the final answer from a generated sequence.
    For example if the generated sequence is:

    5350+9687<assistant_start>
    <start_think>
    =0+7=7<start_carry>0<end_carry>
    =5+8=13<start_carry>1<end_carry>
    =3+6+1=10<start_carry>1<end_carry>
    =5+9+1=15<start_carry>1<end_carry>
    <end_think>
    =15037
    <assistant_end>
    
    We want to extract the 15037
    
    Args:
        text: Generated text from the model
    
    Returns:
        The extracted answer as a string, or empty string if not found
    """
    # Remove <assistant_end> token if present
    text = text.replace('<assistant_end>', '')
    
    # Split by '=' and get the last part
    parts = text.split('=')
    if len(parts) > 0:
        # Get the last part and strip whitespace
        answer = parts[-1].strip()
        # Extract just the digits
        match = re.search(r'\d+', answer)
        if match:
            return match.group()
    
    return ""

def extract_ground_truth(text, reasoning=True):
    """
    Extract the two operands and correct answer from the input text.

    If the text is:

    5350+9687<assistant_start>
    <start_think>
    =0+7=7<start_carry>0<end_carry>
    =5+8=13<start_carry>1<end_carry>
    =3+6+1=10<start_carry>1<end_carry>
    =5+9+1=15<start_carry>1<end_carry>
    <end_think>
    =15037
    <assistant_end>
    
    We want to extract:
        a = 5350
        b = 9687

    and evaluate the true output a + b

    Args:
        text: Original text
    
    Returns:
        Tuple of (operand1, operand2, correct_answer)
    """
    # Extract the initial addition problem (before <start_think>)
    if reasoning:
        problem = text.split('<start_think>')[0].replace("<assistant_start>", "")
    else:
        problem = text.split('=')[0].replace("<assistant_start>", "")

    # Parse "a+b"
    parts = problem.split('+')
    a = int(parts[0].strip())
    b = int(parts[1].strip())
    
    correct_answer = a + b
    
    return a, b, correct_answer

@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    testloader,
    device = "cpu",
    max_new_tokens = 200,
    temperature = 0.3,
    sample = False,
    reasoning=True
):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The GPT model to evaluate
        tokenizer: Tokenizer instance
        testloader: DataLoader for test data
        device: Device to run evaluation on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        sample: Whether to sample or use greedy decoding
        verbose: If True, print incorrect predictions
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    incorrect_examples = []
    
    print("Evaluating model on test set...")
    
    for batch in tqdm(testloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):

            ### Get the original full sequence (for ground truth) the input_ids dont contain the last token (<assistant_end> but we dont)
            ### really care for inference time so we are good!
            full_sequence = tokenizer.decode(input_ids[i])
            
            ### Extract ground truth answers
            a, b, correct_answer = extract_ground_truth(full_sequence, reasoning)
            
            ### Get just the problem part (e.g., "12+34") so we can pass as prompt to LLM
            if reasoning:
                problem_text = full_sequence.split('<start_think>')[0].strip().replace("<assistant_start>", "")
            else:
                problem_text = full_sequence.split('=')[0].strip().replace("<assistant_start>", "")

            ### Encode the problem text 
            problem_tokens = tokenizer.encode(problem_text, is_prompt=True)
            problem_tensor = torch.tensor(problem_tokens).unsqueeze(0).to(device)
            
            ### Generate completion
            generated = model.generate(
                problem_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                sample=sample,
                eos_token_id=tokenizer.assistant_end_id,
            )
            
            ### Decode generated text
            generated_text = tokenizer.decode(generated[0])
            
            ### Extract predicted answer
            predicted_answer = extract_answer(generated_text)
            
            ### Check if correct
            is_correct = predicted_answer == str(correct_answer)
            
            if is_correct:
                correct += 1
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }
 