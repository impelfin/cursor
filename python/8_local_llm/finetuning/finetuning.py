import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import subprocess # GGUF 변환을 위해 필요

# --- 1. 설정 변수 ---
# 파인튜닝할 원본 기본 모델의 로컬 경로.
# 이 경로에 config.json, model.safetensors (또는 pytorch_model.bin), tokenizer.json 등 모든 파일이 있어야 한다.
base_model_id = "./llama3.2-1b" # <<< 실제 환경에 맞게 이 경로를 확인하고 필요시 수정.

sft_json_path = "./sft.json" # 학습 데이터셋 JSON 파일 경로
output_dir = "./llama3.2-1b-finetuned-sft" # 파인튜닝 결과 및 병합 모델 저장 경로

gguf_output_name = "llama3.2-1b-finetuned-sft.gguf"
gguf_output_path = os.path.join(output_dir, gguf_output_name)

# llama.cpp 저장소의 루트 경로.
# convert_hf_to_gguf.py 스크립트가 이 경로 아래에 있어야 한다.
llama_cpp_path = "llama.cpp" # <<< 실제 환경에 맞게 이 경로를 확인하고 필요시 수정.

# --- 2. 모델 및 토크나이저 로드 ---
print(f"모델 로드 중: {base_model_id}...")

# torch_dtype을 float32로 설정하여 안정성을 높인다.
# device_map="auto"는 사용 가능한 디바이스(Mac의 경우 MPS)에 모델을 분배한다.
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float32 # <<< float32로 통일 (가장 안정적)
    )
    model.config.use_cache = False # 학습 중 캐시 비활성화

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    # 패딩 토큰 설정 (모델에 따라 필요)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 패딩 방향 설정 (오른쪽 패딩)

except Exception as e:
    print(f"\n--- 오류: 모델 또는 토크나이저 로드 중 오류 발생 ---")
    print(f"원인: {e}")
    print(f"'{base_model_id}' 경로에 모델 파일이 완전히 존재하는지, 그리고 PyTorch/Transformers 설치가 올바른지 확인하세요.")
    exit(1)

# --- 3. 데이터셋 로드 및 전처리 ---
print(f"데이터셋 로드 중: {sft_json_path}...")
try:
    with open(sft_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list(raw_data)
except FileNotFoundError:
    print(f"오류: {sft_json_path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit(1)
except json.JSONDecodeError:
    print(f"오류: {sft_json_path} 파일이 유효한 JSON 형식이 아닙니다.")
    exit(1)
except Exception as e:
    print(f"데이터셋 로드 중 오류 발생: {e}")
    exit(1)

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
    fp16=False, # <<< False로 유지 (float32 학습을 위함)
    bf16=False, # <<< False로 유지 (float32 학습을 위함)
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
    # tokenizer=tokenizer, # 토크나이저도 명시적으로 전달 (권장)
)

print("\n--- 파인튜닝 시작 ---")
try:
    trainer.train()
    print("\n--- 파인튜닝 완료 ---")
except Exception as e:
    print(f"\n--- 오류: 파인튜닝 중 오류 발생 ---")
    print(f"원인: {e}")
    print("메모리 부족, 학습 인자 설정 오류 등을 확인하세요.")
    exit(1)


# --- 8. 파인튜닝된 LoRA 어댑터와 토크나이저 저장 ---
# SFTTrainer는 학습이 끝나면 자동으로 output_dir에 저장한다.
# 하지만 명시적으로 다시 저장하는 것은 안전하다.
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n파인튜닝된 LoRA 어댑터와 토크나이저가 '{output_dir}'에 저장되었습니다.")


# --- 9. LoRA 어댑터를 기본 모델에 병합 및 전체 모델 저장 (GGUF 변환을 위해) ---
print("\nLoRA 어댑터를 기본 모델에 병합 중 (GGUF 변환 준비)...")

try:
    base_model_full = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        return_dict=True,
        torch_dtype=torch.float32, # <<< float32로 통일
        device_map="auto",
    )

    from peft import PeftModel
    model_to_merge = PeftModel.from_pretrained(base_model_full, output_dir)
    merged_model = model_to_merge.merge_and_unload()

    merged_model_save_path = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_model_save_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_save_path) # 병합된 모델과 토크나이저 함께 저장

    print(f"\n병합된 모델이 '{merged_model_save_path}'에 저장되었습니다. 이제 GGUF로 변환할 수 있습니다.")

except Exception as e:
    print(f"\n--- 오류: LoRA 어댑터 병합 중 오류 발생 ---")
    print(f"원인: {e}")
    print(f"'{base_model_id}' 경로의 기본 모델 파일이 온전한지, 그리고 메모리가 충분한지 확인하세요.")
    exit(1)


# --- 10. GGUF 변환 (llama.cpp의 convert_hf_to_gguf.py 사용) ---
convert_py_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")

if not os.path.exists(convert_py_path):
    print(f"\n경고: '{convert_py_path}'를 찾을 수 없습니다.")
    print("GGUF 변환을 위해서는 llama.cpp 레포지토리와 해당 스크립트가 필요합니다.")
    print("위의 '스크립트 실행 전 필수 확인 사항'을 다시 확인하세요.")
    exit(1)
else:
    print(f"\nGGUF 변환 시작: {gguf_output_path}...")
    # --outtype F32로 설정하여 GGUF 파일도 float32로 변환 (안정성 증대)
    convert_command = [
        "python", convert_py_path,
        merged_model_save_path, # 병합된 모델 경로 사용
        "--outfile", gguf_output_path,
        "--outtype", "F32" # <<< GGUF도 float32로 변환 (가장 안정적)
    ]
    print(f"실행 명령: {' '.join(convert_command)}")

    try:
        # subprocess.run을 shell=True 없이 리스트 형태로 사용 (더 안전함)
        process = subprocess.run(convert_command, check=True, capture_output=True, text=True)
        print("GGUF 변환 출력:")
        print(process.stdout)
        if process.stderr:
            print("GGUF 변환 경고/오류 (있는 경우):")
            print(process.stderr)

    except subprocess.CalledProcessError as e:
        print(f"\n--- 오류: GGUF 변환 중 오류 발생 ---")
        print(f"명령어: {' '.join(e.cmd)}")
        print(f"반환 코드: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        print("llama.cpp가 빌드되었고, 변환 스크립트 경로가 올바른지 확인하세요.")
        exit(1)
    except FileNotFoundError:
        print(f"\n--- 오류: 'python' 명령어를 찾을 수 없습니다. Python이 PATH에 추가되었는지 확인하세요. ---")
        exit(1)
    except Exception as e:
        print(f"\n--- 예상치 못한 오류: {e} ---")
        exit(1)

    if os.path.exists(gguf_output_path):
        print(f"\n--- 성공: 파인튜닝된 GGUF 모델이 '{gguf_output_path}'에 생성되었습니다. ---")
        print("이제 이 GGUF 파일을 사용하여 Ollama에 모델을 등록하고 실행할 수 있습니다.")
        print("\n--- 다음 단계 ---")
        print(f"1. Ollama bash 스크립트에서 GGUF_MODEL_PATH가 '{gguf_output_path}'를 정확히 가리키는지 확인하세요.")
        print(f"2. Ollama bash 스크립트를 다시 실행하여 모델을 등록하고 실행하세요.")
    else:
        print("\n--- 오류: GGUF 모델 생성에 실패했습니다. llama.cpp 변환 과정을 다시 확인해주세요. ---")
        exit(1)