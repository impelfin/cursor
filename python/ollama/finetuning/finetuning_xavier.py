import os
import json
import torch
import sys
import subprocess
import logging
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# =========================
# 1. 경로 및 환경 변수 설정
# =========================
logger.info("1단계: 경로 및 환경 변수 설정")
# 모델을 Qwen1.5-0.5B-Chat으로 변경합니다.
base_model_local_path = "Qwen/Qwen1.5-0.5B-Chat"
sft_json_path = "./sft.json"

# 출력 디렉토리 이름도 변경된 모델에 맞춰 변경합니다.
output_dir = "./finetuned-qwen-0.5b"
os.makedirs(output_dir, exist_ok=True)

gguf_output_name = f"{os.path.basename(base_model_local_path).replace('/', '-')}-finetuned.gguf"
gguf_output_path = os.path.join(output_dir, gguf_output_name)
llama_cpp_path = "/home/moon/work/cursor/python/ollama/finetuning/llama.cpp" # 정확한 절대 경로로 수정해주세요!

# llama.cpp 경로를 sys.path에 추가하여 convert_hf_to_gguf 모듈을 import 할 수 있도록 합니다.
sys.path.append(llama_cpp_path)
try:
    # convert_hf_to_gguf.py를 import합니다.
    import convert_hf_to_gguf as llama_converter
except ImportError as e:
    logger.error(f"convert_hf_to_gguf.py를 임포트할 수 없습니다. llama.cpp 경로 확인 또는 파일명 확인: {e}")
    sys.exit(1)


# =========================
# 2. 디바이스 확인 및 설정
# =========================
logger.info("2단계: CUDA 디바이스 확인")
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")
else:
    logger.error("CUDA 사용 불가. Jetson에서 CPU로는 학습이 사실상 불가능합니다.")
    sys.exit(1)

# =========================
# 3. 모델 및 토크나이저 로드
# =========================
logger.info("3단계: 모델 및 토크나이저 로드")
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_local_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=False
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable() # 메모리 절약

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_local_path,
        trust_remote_code=True,
        local_files_only=False,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("모델/토크나이저 로드 성공.")
except Exception as e:
    logger.error(f"모델/토크나이저 로드 오류: {e}")
    sys.exit(1)

# =========================
# 4. 데이터셋 로드 및 전처리 (10개만 사용)
# =========================
logger.info("4단계: 데이터셋 로드 및 전처리 (10개 샘플만 사용)")
try:
    with open(sft_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    # 여기서 처음 10개만 사용하도록 데이터를 슬라이싱합니다.
    raw_data = raw_data[:10]

    def format_data_for_sft(example):
        if "instruction" in example and "output" in example:
            text = f"### Instruction:\n{example['instruction']}\n### Output:\n{example['output']}"
        elif "prompt" in example and "completion" in example:
            text = f"### Prompt:\n{example['prompt']}\n### Completion:\n{example['completion']}"
        elif "text" in example:
            text = example["text"]
        else:
            logger.error(f"데이터셋 샘플에 필수 필드 없음: {example}")
            sys.exit(1)
        return {"text": text}

    dataset = Dataset.from_list(raw_data).map(format_data_for_sft, remove_columns=list(raw_data[0].keys()) if raw_data else [])
    logger.info(f"데이터셋 로드 및 포맷팅 성공. {len(dataset)}개 샘플.")
except Exception as e:
    logger.error(f"데이터셋 로드/전처리 오류: {e}")
    sys.exit(1)

# =========================
# 5. LoRA 설정 (최소화)
# =========================
logger.info("5단계: LoRA 설정(최소화)")
peft_config = LoraConfig(
    lora_alpha=4,
    lora_dropout=0.05,
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
    # Qwen 모델에 더 적합한 target_modules (대부분의 Llama/Qwen 계열에서 잘 작동)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# =========================
# 6. LoRA 적용
# =========================
logger.info("6단계: LoRA 어댑터 적용")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
logger.info("LoRA 어댑터 적용 완료.")

# =========================
# 7. 학습 인자 설정 (최소화)
# =========================
logger.info("7단계: 학습 인자 설정(최소화)")
sft_training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="adafactor",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    save_steps=50,
    logging_steps=10,
    push_to_hub=False,
    report_to="none",
    fp16=True,
    bf16=False, # Jetson Nano는 bf16 지원하지 않습니다.
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    disable_tqdm=False,
    max_seq_length=32,
    dataset_text_field="text",
    packing=False,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    auto_find_batch_size=False,
)

# =========================
# 8. SFTTrainer 설정 및 학습
# =========================
logger.info("8단계: SFTTrainer 설정 및 파인튜닝 시작")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_training_args,
    tokenizer=tokenizer,
)

try:
    trainer.train()
    logger.info("파인튜닝 완료")
except Exception as e:
    logger.error(f"파인튜닝 중 오류: {e}")
    sys.exit(1)

# =========================
# 9. 파인튜닝 결과 저장
# =========================
logger.info("9단계: 파인튜닝 결과 저장")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"LoRA 어댑터와 토크나이저 저장 완료: {output_dir}")

# =========================
# 10. LoRA 어댑터 병합 및 전체 모델 저장
# =========================
logger.info("10단계: LoRA 어댑터 병합 및 전체 모델 저장")
try:
    base_model_full = AutoModelForCausalLM.from_pretrained(
        base_model_local_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=False
    ).to(device)

    model_to_merge = PeftModel.from_pretrained(base_model_full, output_dir)
    merged_model = model_to_merge.merge_and_unload()

    merged_model_save_path = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_model_save_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_save_path)

    logger.info(f"병합된 모델 저장 완료: {merged_model_save_path}")

except Exception as e:
    logger.error(f"LoRA 병합 오류: {e}")
    sys.exit(1)

# =========================
# 10.5단계: model.safetensors를 pytorch_model.bin으로 변환 (선택 사항, 필요 시 활성화)
# convert_hf_to_gguf.py가 safetensors도 처리 가능하지만, 안정성을 위해 bin도 생성해둡니다.
# =========================
# logger.info("10.5단계: model.safetensors를 pytorch_model.bin으로 변환 시작")
# pytorch_model_path_for_gguf_temp = os.path.join(merged_model_save_path, "pytorch_model.bin")
# try:
#     model_for_bin_conversion = AutoModelForCausalLM.from_pretrained(
#         merged_model_save_path, # safetensors가 있는 디렉토리
#         torch_dtype=torch.float16,
#         device_map="auto" # GPU 사용
#     )
#     model_for_bin_conversion.save_pretrained(merged_model_save_path, safe_serialization=False)
#     logger.info(f"model.safetensors가 {pytorch_model_path_for_gguf_temp}로 성공적으로 변환되었습니다.")
# except Exception as e:
#     logger.error(f"model.safetensors 변환 오류: {e}")
#     sys.exit(1)


# =========================
# 11. GGUF 변환 (convert_hf_to_gguf.py를 직접 임포트하여 사용)
# =========================
logger.info("11단계: GGUF 변환 시작 (convert_hf_to_gguf.py 직접 임포트)")
logger.info(f"GGUF 변환 시작: {gguf_output_path}")

try:
    # llama_converter.main 함수를 호출합니다.
    # convert_hf_to_gguf.py 스크립트의 main 함수는 인자를 파싱하는 로직을 가지고 있을 겁니다.
    # sys.argv를 임시로 조작하여 인자를 전달합니다.
    original_argv = sys.argv
    sys.argv = [
        "convert_hf_to_gguf.py", # 스크립트 이름 (첫 번째 인자)
        merged_model_save_path,  # 입력 모델 디렉토리
        "--outfile", gguf_output_path,
        "--outtype", "f16"
    ]

    llama_converter.main() # convert_hf_to_gguf.py 스크립트의 main 함수 호출

    sys.argv = original_argv # sys.argv를 원래대로 복원

    logger.info("GGUF 변환 완료")

except Exception as e:
    logger.error(f"GGUF 변환 오류 (직접 임포트 방식): {e}")
    sys.exit(1)

# =========================
# 12. 학습 완료
# =========================
logger.info("12단계: 학습 완료")
