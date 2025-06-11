from huggingface_hub import snapshot_download

model_id = 'meta-llama/Llama-3.2-1B'
snapshot_download(repo_id=model_id, local_dir='/Users/Lune/Documents/GitHub/full/ai-2/8_local_llm/finetuning/', local_dir_use_symlinks=False, revision='main')