import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 4bit model but compute in bf16
model_4b_cd_bf16 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)


# 8bit model but compute in bf16
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)

model_8b_cd_bf16 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)


nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    #bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)


print(f"Default model size : {model.get_memory_footprint()/1e+6} MB")

print(f"8 bit model size : {model_8b_cd_bf16.get_memory_footprint()/1e+6} MB")

print(f"4 bit model size : {model_4b_cd_bf16.get_memory_footprint()/1e+6} MB")

print(f"nf4 model size : {model_nf4.get_memory_footprint()/1e+6} MB")



# Load a small portion of the Wikipedia dataset
num_samples = 1500
wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
wiki_data = wiki_dataset.take(num_samples)
wiki_data = list(wiki_data)  # Convert to a list for easier processing

def compute_perplexity(model, encodings):
    max_length = 2048 # model.config.n_positions
    stride = max_length # 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()



# Compute perplexity for each article
perplexities = []

for article in tqdm(wiki_data):
    # Tokenize the entire article
    encodings = tokenizer(article['text'], return_tensors='pt', truncation=True, max_length=2048)#model.config.n_positions)

    # Compute perplexity for the article
    perplexity = compute_perplexity(model, encodings)
    perplexities.append(perplexity)

# Print perplexities for each article
# for i, (article, perplexity) in enumerate(zip(wiki_data, perplexities)):
#     print(f"Article {i+1}: {article['text'][:100]}...")  # Print first 100 characters of the article
#     print(f"Perplexity: {perplexity:.4f}\n")

# Calculate and print average perplexity across all articles
average_perplexity = sum(perplexities) / len(perplexities)
print(f"Overall Average Perplexity of Default Model: {average_perplexity:.4f}")

model.to("cpu")





# Compute perplexity for each article
perplexities = []

for article in tqdm(wiki_data):
    # Tokenize the entire article
    encodings = tokenizer(article['text'], return_tensors='pt', truncation=True, max_length=2048)#model.config.n_positions)

    # Compute perplexity for the article
    perplexity = compute_perplexity(model_8b_cd_bf16, encodings)
    perplexities.append(perplexity)

# Print perplexities for each article
# for i, (article, perplexity) in enumerate(zip(wiki_data, perplexities)):
#     print(f"Article {i+1}: {article['text'][:100]}...")  # Print first 100 characters of the article
#     print(f"Perplexity: {perplexity:.4f}\n")

# Calculate and print average perplexity across all articles
average_perplexity = sum(perplexities) / len(perplexities)
print(f"Overall Average Perplexity of 8 bit Model: {average_perplexity:.4f}")








# Compute perplexity for each article
perplexities = []

for article in tqdm(wiki_data):
    # Tokenize the entire article
    encodings = tokenizer(article['text'], return_tensors='pt', truncation=True, max_length=2048)#model.config.n_positions)

    # Compute perplexity for the article
    perplexity = compute_perplexity(model_4b_cd_bf16, encodings)
    perplexities.append(perplexity)

# Print perplexities for each article
# for i, (article, perplexity) in enumerate(zip(wiki_data, perplexities)):
#     print(f"Article {i+1}: {article['text'][:100]}...")  # Print first 100 characters of the article
#     print(f"Perplexity: {perplexity:.4f}\n")

# Calculate and print average perplexity across all articles
average_perplexity = sum(perplexities) / len(perplexities)
print(f"Overall Average Perplexity of 4 bit Model: {average_perplexity:.4f}")









# Compute perplexity for each article
perplexities = []

for article in tqdm(wiki_data):
    # Tokenize the entire article
    encodings = tokenizer(article['text'], return_tensors='pt', truncation=True, max_length=2048)#model.config.n_positions)

    # Compute perplexity for the article
    perplexity = compute_perplexity(model_4b_cd_bf16, encodings)
    perplexities.append(perplexity)

# Print perplexities for each article
# for i, (article, perplexity) in enumerate(zip(wiki_data, perplexities)):
#     print(f"Article {i+1}: {article['text'][:100]}...")  # Print first 100 characters of the article
#     print(f"Perplexity: {perplexity:.4f}\n")

# Calculate and print average perplexity across all articles
average_perplexity = sum(perplexities) / len(perplexities)
print(f"Overall Average Perplexity of nf4 bit Model: {average_perplexity:.4f}")