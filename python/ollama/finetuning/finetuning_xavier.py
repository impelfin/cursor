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
base_model_local_path = "facebook/opt-125m"
sft_json_path = "./sft.json"  # 10개 이하 소규모 데이터셋 사용 권장
output_dir = "./finetuned-opt-125m" # 출력 디렉토리 이름도 변경
os.makedirs(output_dir, exist_ok=True)
gguf_output_name = f"{os.path.basename(base_model_local_path).replace('/', '-')}-finetuned.gguf"
gguf_output_path = os.path.join(output_dir, gguf_output_name)
llama_cpp_path = "llama.cpp"

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
    model.gradient_checkpointing_enable()  # 메모리 절약

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_local_path,
        trust_remote_code=True,
        local_files_only=False, 
        use_fast=False # OpenLLaMA는 fast tokenizer 사용 시 문제가 있을 수 있음
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
    raw_data = raw_data[:10]  # 10개만 추출

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

    dataset = Dataset.from_list(raw_data).map(format_data_for_sft, remove_columns=list(raw_data[0].keys()))
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
    target_modules=["q_proj", "v_proj"], # VRAM 문제 해결을 위해 다시 최소화된 타겟 모듈로 복귀
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
    gradient_accumulation_steps=16, # 그래디언트 누적 단계 유지 (최대화)
    optim="adafactor", # adafactor 유지
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    save_steps=50,
    logging_steps=10,
    push_to_hub=False,
    report_to="none",
    fp16=True,
    bf16=False,
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
    # resume_from_checkpoint=True, # 기존 체크포인트에서 학습 재개
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
    # 병합을 위해 다시 기본 모델 로드 (Hugging Face Hub에서 다운로드)
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
# 11. GGUF 변환
# =========================
logger.info("11단계: GGUF 변환 시작")
logger.info(f"GGUF 변환 시작: {gguf_output_path}")

# merged_model_file_path = os.path.join(merged_model_save_path, "pytorch_model.bin")
merged_model_file_path = os.path.join(merged_model_save_path, "model.safetensors")

convert_command = [
    f"{sys.executable}", # 현재 파이썬 인터프리터 사용
    os.path.join(llama_cpp_path, "convert-refact-hf-to-gguf.py")
    "--outfile", gguf_output_path,
    "--outtype", "f16"
]

logger.info(f"실행 명령: {' '.join(convert_command)}")

try:
    subprocess.run(convert_command, check=True, cwd=os.getcwd()) # cwd 명시적으로 현재 작업 디렉토리
    logger.info("GGUF 변환 완료")
except subprocess.CalledProcessError as e:
    logger.error(f"GGUF 변환 오류: {e}")
    sys.exit(1)

# =========================
# 12. 학습 완료
# =========================
logger.info("12단계: 학습 완료")
