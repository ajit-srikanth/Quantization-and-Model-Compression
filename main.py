import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# model_id = "Salesforce/codegen-350M-mono"
# model_id = "microsoft/phi-1_5"
# model_id = "alpindale/Llama-3.2-1B-Instruct"
# model_id = "EleutherAI/gpt-neo-1.3B"
# model_id = "facebook/opt-1.3b"
model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
# print(model)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
#print(tokenizer.model_max_length)
#print(model.config.max_position_embeddings)
#exit(0)

# find footprint before quantization
memory_footprint_before_quantization = model.get_memory_footprint()/1e+6
print(f"model size before quantization : {memory_footprint_before_quantization} MB")

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
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
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
print(f"Overall Average Perplexity before quantization: {average_perplexity:.4f}")


print(model)

# W8: Weights will be stored in 8 bits (int8).
# A16: Activations will be stored in 16 bits (FP16). lol but we use fp32 here

class W8A16LinearLayer(nn.Module):
  def __init__(self, input_features, output_features, bias=True, dtype=torch.float32):
    super().__init__()

    self.register_buffer("int8_weights", torch.randint(-128,127, (output_features, input_features), dtype=torch.int8))
    self.register_buffer("scales", torch.randn((output_features), dtype= dtype))

    if bias:
      self.register_buffer("bias", torch.randn((1, output_features), dtype = dtype))
    else:
      self.bias = None

  def forward(self, inputs):
    converted_weights = self.int8_weights.to(inputs.dtype)
    output = F.linear(inputs, converted_weights) * self.scales

    if self.bias is not None:
      output = output + self.bias

    return output

  def quantize(self, weights):
    w_fp32 = weights.clone().to(torch.float32)

    scales = w_fp32.abs().max(dim=-1).values/127
    scales = scales.to(weights.dtype)

    int8_weights = torch.round(weights/scales.unsqueeze(1)).to(torch.int8)

    self.int8_weights  = int8_weights
    self.scales = scales

def replace_linear_layer_with_W8A16Linear_layer_and_quantization(module, target , exclude_list):
  for name, child in module.named_children():
    if isinstance(child, nn.Linear) and not any([x == name for x in exclude_list]):
      old_bias = child.bias
      old_weights = child.weight

      new_module = target(child.in_features, child.out_features,
                                   old_bias is not None, child.weight.dtype)

      setattr(module, name, new_module)
      getattr(module, name).quantize(old_weights)

      if old_bias is not None:
        getattr(module, name).bias = old_bias

    else:
      replace_linear_layer_with_W8A16Linear_layer_and_quantization(child, target, exclude_list)


def replace_few_linear_layer_with_W8A16Linear_layer_and_quantization(module, target, exclude_list, max_layers=None, current_count=0):
    if max_layers is not None and current_count >= max_layers:
        return current_count

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in exclude_list]):
            if max_layers is None or current_count < max_layers:
                old_bias = child.bias
                old_weights = child.weight

                new_module = target(child.in_features, child.out_features,
                                    old_bias is not None, child.weight.dtype)

                setattr(module, name, new_module)
                getattr(module, name).quantize(old_weights)

                if old_bias is not None:
                    getattr(module, name).bias = old_bias

                current_count += 1
                if max_layers is not None and current_count >= max_layers:
                    return current_count
        else:
            current_count = replace_few_linear_layer_with_W8A16Linear_layer_and_quantization(
                child, target, exclude_list, max_layers, current_count)
            if max_layers is not None and current_count >= max_layers:
                return current_count

    return current_count

# Usage: Replace only the first N layers (e.g., first 5 layers)
# num_layers_to_quantize = 6
# replace_few_linear_layer_with_W8A16Linear_layer_and_quantization(
#     model, W8A16LinearLayer, ["lm_head"], max_layers=num_layers_to_quantize)

replace_linear_layer_with_W8A16Linear_layer_and_quantization(
    model, W8A16LinearLayer, ["lm_head"])

print(model)

memory_footprint_after_quantization = model.get_memory_footprint()/1e+6
print(f"model size after quantization : {np.round(memory_footprint_after_quantization,2)} MB")


print(f"Memory saved : {np.round((memory_footprint_before_quantization - memory_footprint_after_quantization), 2)} MB")


# Compute perplexity for each article
perplexities = []

for article in tqdm(wiki_data):
    # Tokenize the entire article
    encodings = tokenizer(article['text'], return_tensors='pt', truncation=True, max_length= 2048) #model.config.n_positions)

    # Compute perplexity for the article
    perplexity = compute_perplexity(model, encodings)
    perplexities.append(perplexity)

# Print perplexities for each article
# for i, (article, perplexity) in enumerate(zip(wiki_data, perplexities)):
#     print(f"Article {i+1}: {article['text'][:100]}...")  # Print first 100 characters of the article
#     print(f"Perplexity: {perplexity:.4f}\n")

# Calculate and print average perplexity across all articles
average_perplexity = sum(perplexities) / len(perplexities)
print(f"Overall Average Perplexity after quantization: {average_perplexity:.4f}")