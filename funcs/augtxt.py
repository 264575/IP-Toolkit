# augtxt.py
# license: see LICENSE file
import os
from collections import Counter
from typing import List, Union, Any
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import streamlit as st

# Initialize GPT-2 model and tokenizer
model_name = "gpt2"
model_gpt = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def settexts(read_paths: List[Union[str, Any]]):
    """
    Setting Input Texts
    """

    input_texts = []
    original_file_names = []
    for uploaded_file in read_paths:
        try:
            text = uploaded_file.read().decode()
            uploaded_file.seek(0)
        except AttributeError:
            with open(uploaded_file, 'r') as f:
                text = f.read()

        original_file_name = os.path.basename(uploaded_file).split('.')[0]
        original_file_names.append(original_file_name)
        input_texts.append(text)

    return input_texts, original_file_names

def generate_texts(prompts, max_length=100, num_return_sequences=1):
    """
    Generating Text by GPT2
    """

    input_ids = [tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]
    input_ids = torch.cat(input_ids, dim=0)

    with torch.no_grad():
        output = model_gpt.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

    generated_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    return generated_texts

def augprompt(batch_size, max_length, read_paths, write_path):

    input_texts, original_file_names = settexts(read_paths)

    total_files = len(input_texts)
    total_batches = (total_files + batch_size - 1) // batch_size

    for i in range(total_batches):
        print(f"Processing batch {i + 1}/{total_batches}")

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_files)
        batch_texts = input_texts[start_idx:end_idx]
        original_names_batch = original_file_names[start_idx:end_idx]

        generated_texts_batch = generate_texts(batch_texts, max_length=max_length, num_return_sequences=1)

        for original_name, generated_text in zip(original_names_batch, generated_texts_batch):
            new_file_path = os.path.join(write_path, f"{original_name}_aug.txt")

            with open(new_file_path, 'w') as new_file:
                new_file.write(generated_text)

def remove_tags_from_files(directory_path, ext, write_path, removal_words_file):
    """
    Removes specified words from all text files in the given directory.

    Parameters:
    - directory_path: Path to the directory containing text files.
    - removal_words_file: Path to the text file containing words to be removed.
    """

    # Reading the words to remove from the words file
    with open(removal_words_file, 'r', encoding='utf-8') as file:
        words_to_remove = {line.strip() for line in file}

    # Iterating through each file in the input directory
    for filename in os.listdir(directory_path):
        if filename.endswith("."+ext):  # assuming text files
            input_file_path = os.path.join(directory_path, filename)
            output_file_path = os.path.join(write_path, filename)

            # Reading the input file
            with open(input_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Processing each line
            updated_lines = []
            for line in lines:
                tokens = line.strip().split(',')
                # Removing tokens that contain any of the specified words as a part of the token
                filtered_tokens = [token for token in tokens if not any(word in token for word in words_to_remove)]
                updated_lines.append(','.join(filtered_tokens))

            # Writing the updated content to the output file
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(updated_lines))

def find_files_with_tokens_or_words(directory, tokens_words_file_path, ext):
    # Reading the tokens or words to find from the specified file
    with open(tokens_words_file_path, 'r', encoding='utf-8') as file:
        tokens_words_to_find = {line.strip() for line in file}

    # Iterating through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith("."+ext):  # assuming text files
            file_path = os.path.join(directory, filename)

            # Reading the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Checking for tokens and words
            tokens = content.split(',')  # Splitting by token delimiter
            words = content.split()      # Splitting by word delimiter

            # If any of the tokens or words are found, print the file name
            for token in tokens:
                if token in tokens_words_to_find: 
                    print(f"token:{token} {filename}")
            for word in words:
                if word in tokens_words_to_find: 
                    print(f"word:{word} {filename}")

def edit_text_files(read_paths, write_path, undesired_tokens_path):
            undesired_tokens = []
            with open(undesired_tokens_path, 'r') as f:
                for text in f.readline():
                    undesired_tokens.append( text.replace(',','') )

            file_paths = os.listdir(read_paths)

            for file_path in file_paths:
                with open(file_path, 'r') as f:
                    text = f.read()

                text = text.split(', ').split(',')

                for index, undesired_token in enumerate(undesired_tokens):
                    if undesired_token in text:
                        pos_index = text.index(undesired_token)
                        text = text.pop(pos_index)

                with open(write_path, 'w') as f:
                    f.write(text)

def print_top_counts(counter, top_n=250):
    """ Prints the top N counts from a Counter object in a more readable format. """
    for item, count in counter.most_common(top_n):
        print(f"{item}: {count}")

def count_tokens_and_words(directory, ext):
    token_counts = Counter()
    word_counts = Counter()

    for filename in os.listdir(directory):
        if filename.endswith("."+ext):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    tokens = line.strip().split(',')
                    token_counts.update(tokens)
                    for token in tokens:
                        words = token.split()
                        word_counts.update(words)

    print("Top Token Counts:")
    print_top_counts(tokens)
    print("Total Tokens:", sum(tokens.values()))

    print("\nTop Word Counts:")
    print_top_counts(words)
    print("Total Words:", sum(words.values()))

def augtxt(batch_size, source_option, read_paths, write_path, op_sel, max_length, undesired_tokens_path):
#def augtxt(batch_size, read_paths, write_path, max_length=100):
    """
    Augment texts
    """
    
    ext = source_option['ext']

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    try:

        if op_sel == 'Augment Texts':
            augprompt(batch_size, max_length, read_paths, write_path)

        elif op_sel == 'Remove Tags':
            remove_tags_from_files(read_paths, ext, write_path, undesired_tokens_path)

        elif op_sel == 'Find File':
            tokens_words_file_path = source_option['tokens_words_file_path']
            find_files_with_tokens_or_words(read_paths, tokens_words_file_path, ext)

        elif op_sel == 'Edit Texts':
            edit_text_files(read_paths, write_path, undesired_tokens_path)

        elif op_sel == 'Stat Texts':
            count_tokens_and_words(read_paths, ext)

    except Exception as e:
        print(f"An error occurred: {e}")
