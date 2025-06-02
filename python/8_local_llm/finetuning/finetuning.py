import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# --- 1. 설정 변수 ---
base_model_id = "./llama3.2-1b"
sft_json_path = "./sft.json"
output_dir = "./llama3.2-1b-finetuned-sft"
gguf_output_name = "llama3.2-1b-finetuned-sft.gguf"
gguf_output_path = os.path.join(output_dir, gguf_output_name)

# --- 2. 모델 및 토크나이저 로드 ---
print(f"모델 로드 중: {base_model_id}...")

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16 # fp16/bf16을 False로 했으므로, torch_dtype은 float16으로 유지
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. 데이터셋 로드 및 전처리 ---
print(f"데이터셋 로드 중: {sft_json_path}...")
try:
    with open(sft_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list(raw_data)
except FileNotFoundError:
    print(f"오류: {sft_json_path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()
except json.JSONDecodeError:
    print(f"오류: {sft_json_path} 파일이 유효한 JSON 형식이 아닙니다.")
    exit()
except Exception as e:
    print(f"데이터셋 로드 중 오류 발생: {e}")
    exit()

# --- 4. LoRA 설정 ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# --- 5. 모델 준비 (LoRA 적용) ---
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 6. 학습 인자 설정 ---
sft_training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_steps=50,
    logging_steps=10,
    push_to_hub=False,
    report_to="none",
    fp16=False, # <-- False로 변경
    bf16=False, # <-- False로 변경
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    disable_tqdm=False,
    max_seq_length=512,
    dataset_text_field="text",
    packing=False,
)

# --- 7. SFTTrainer 설정 및 학습 시작 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_training_args,
)

print("\n--- 파인튜닝 시작 ---")
trainer.train()
print("\n--- 파인튜닝 완료 ---")

# --- 8. 파인튜닝된 모델 저장 ---
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n파인튜닝된 LoRA 어댑터와 토크나이저가 '{output_dir}'에 저장되었습니다.")

# --- 9. LoRA 어댑터를 기본 모델에 병합 및 전체 모델 저장 (GGUF 변환을 위해) ---
print("\nLoRA 어댑터를 기본 모델에 병합 중 (GGUF 변환 준비)...")

base_model_full = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

from peft import PeftModel
model_to_merge = PeftModel.from_pretrained(base_model_full, output_dir)
merged_model = model_to_merge.merge_and_unload()

merged_model_save_path = os.path.join(output_dir, "merged_model")
merged_model.save_pretrained(merged_model_save_path, safe_serialization=True)
tokenizer.save_pretrained(merged_model_save_path)

print(f"\n병합된 모델이 '{merged_model_save_path}'에 저장되었습니다. 이제 GGUF로 변환할 수 있습니다.")

# --- 10. GGUF 변환 (llama.cpp의 convert_hf_to_gguf.py 사용) ---
llama_cpp_path = "llama.cpp"
convert_py_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")

if not os.path.exists(convert_py_path):
    print(f"\n경고: '{convert_py_path}'를 찾을 수 없습니다.")
    print("GGUF 변환을 위해서는 llama.cpp 레포지토리가 필요합니다.")
    print("다음 명령으로 llama.cpp를 클론하세요: git clone https://github.com/ggerganov/llama.cpp.git")
    print("그리고 llama.cpp/requirements.txt를 설치하세요: pip install -r llama.cpp/requirements.txt")
    print("또는 해당 스크립트를 직접 복사하여 사용하세요.")
else:
    print(f"\nGGUF 변환 시작: {gguf_output_path}...")
    convert_command = f"python {convert_py_path} {merged_model_save_path} --outfile {gguf_output_path} --outtype F16"
    print(f"실행 명령: {convert_command}")

    import subprocess
    try:
        subprocess.run(convert_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n--- 오류: GGUF 변환 중 오류 발생. 자세한 내용은 위 에러 메시지를 확인하세요. ---")
        print(f"명령어: {e.cmd}")
        print(f"반환 코드: {e.returncode}")
        exit(1)

    if os.path.exists(gguf_output_path):
        print(f"\n--- 성공: 파인튜닝된 GGUF 모델이 '{gguf_output_path}'에 생성되었습니다. ---")
        print("이제 이 GGUF 파일을 사용하여 Ollama에 모델을 등록하고 실행할 수 있습니다.")
    else:
        print("\n--- 오류: GGUF 모델 생성에 실패했습니다. llama.cpp 변환 과정을 확인해주세요. ---")