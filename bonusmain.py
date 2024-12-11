import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, upload_folder

def convert_to_ggml(model_id, output_dir):
    # Download the model from Hugging Face
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Save the model locally
    local_model_path = "temp_model"
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)

    # Run the conversion script
    conversion_script = "path/to/llama.cpp/convert.py"  # Update this path
    subprocess.run([
        "python", conversion_script,
        local_model_path,
        "--outfile", f"{output_dir}/model.ggml.bin",
        "--outtype", "f16"
    ])

    # Clean up temporary files
    subprocess.run(["rm", "-rf", local_model_path])

def run_ggml_model(model_path, prompt):
    llama_cpp_path = "path/to/llama.cpp/main"  # Update this path
    result = subprocess.run([
        llama_cpp_path,
        "-m", model_path,
        "-p", prompt
    ], capture_output=True, text=True)
    return result.stdout

def push_to_hub(local_dir, repo_id, token):
    api = HfApi()
    api.create_repo(repo_id, private=False, token=token)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )

if __name__ == "__main__":
    # Set up parameters
    model_id = "facebook/opt-350m"
    output_dir = "ggml_model"
    ggml_model_path = f"{output_dir}/model.ggml.bin"
    prompt = "Once upon a time"
    hub_token = "your_huggingface_token"  # Replace with your actual token
    repo_id = "your-username/ggml-opt-350m"  # Replace with your desired repo name

    # Convert the model
    convert_to_ggml(model_id, output_dir)

    # Run the model locally
    output = run_ggml_model(ggml_model_path, prompt)
    print("Model output:", output)

    # Push to Hugging Face Hub
    push_to_hub(output_dir, repo_id, hub_token)
    print(f"Model pushed to {repo_id} on Hugging Face Hub")