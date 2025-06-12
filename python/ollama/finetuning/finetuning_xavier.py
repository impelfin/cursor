import os
import json
import torch
import sys
import subprocess
import logging
from datetime import datetime

# 필수 라이브러리: transformers, peft, datasets, accelerate, trl
# bitsandbytes는 설치/임포트하지 않습니다!

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

logger.info("라이브러리 임포트 및 로깅 설정 완료.")

# --- 1. 설정 변수 ---
base_model_local_path = "./llama-3.2-1b"
sft_json_path = "./sft.json"
output_dir = "./finetuned-llama-3.2-1b"
os.makedirs(output_dir, exist_ok=True)
gguf_output_name = f"{os.path.basename(base_model_local_path).lower()}-finetuned.gguf"
gguf_output_path = os.path.join(output_dir, gguf_output_name)
llama_cpp_path = "llama.cpp"

logger.info("설정 변수 로드 완료.")

# --- 2. 모델 및 토크나이저 로드 ---
logger.info(f"모델 로드 중: {base_model_local_path}...")

if torch.cuda.is_available():
    device = "cuda"
    logger.info(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")
else:
    device = "cpu"
    logger.error("CUDA 사용 불가. CPU로는 Jetson에서 학습이 사실상 불가능합니다.")
    sys.exit(1)

try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_local_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_local_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("모델/토크나이저 로드 성공.")

except Exception as e:
    logger.error(f"모델/토크나이저 로드 오류: {e}")
    sys.exit(1)

# --- 3. 데이터셋 로드 및 전처리 ---
logger.info(f"데이터셋 로드 중: {sft_json_path}...")
try:
    with open(sft_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

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
logger.info("LoRA 어댑터 적용 완료.")

# --- 6. 학습 인자 설정 ---
sft_training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,
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
    max_seq_length=128,
    dataset_text_field="text",
    packing=False,
)

# --- 7. SFTTrainer 설정 및 학습 시작 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_training_args,
    tokenizer=tokenizer,
)

logger.info("파인튜닝 시작")
try:
    trainer.train()
    logger.info("파인튜닝 완료")
except Exception as e:
    logger.error(f"파인튜닝 중 오류: {e}")
    sys.exit(1)

# --- 8. 파인튜닝 결과 저장 ---
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"LoRA 어댑터와 토크나이저 저장 완료: {output_dir}")

# --- 9. LoRA 어댑터 병합 및 전체 모델 저장 ---
logger.info("LoRA 어댑터 병합 중...")

try:
    base_model_full = AutoModelForCausalLM.from_pretrained(
        base_model_local_path,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        offload_buffers=True
    )

    model_to_merge = PeftModel.from_pretrained(base_model_full, output_dir)
    merged_model = model_to_merge.merge_and_unload()

    merged_model_save_path = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_model_save_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_save_path)

    logger.info(f"병합된 모델 저장 완료: {merged_model_save_path}")

except Exception as e:
    logger.error(f"LoRA 병합 오류: {e}")
    sys.exit(1)

# --- 10. GGUF 변환 ---
convert_py_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")

if not os.path.exists(convert_py_path):
    logger.warning(f"'{convert_py_path}'를 찾을 수 없습니다. llama.cpp 클론/빌드 필요.")
    sys.exit(1)
else:
    logger.info(f"GGUF 변환 시작: {gguf_output_path}")
    python_executable = sys.executable

    convert_command = [
        python_executable,
        convert_py_path,
        merged_model_save_path,
        "--outfile", gguf_output_path,
        "--outtype", "f16"
    ]
    logger.info(f"실행 명령: {' '.join(convert_command)}")

    try:
        process = subprocess.run(convert_command, check=True)
    except Exception as e:
        logger.error(f"GGUF 변환 오류: {e}")
        sys.exit(1)

    if os.path.exists(gguf_output_path):
        logger.info(f"GGUF 모델 생성 완료: {gguf_output_path}")
    else:
        logger.error("GGUF 모델 생성 실패")
        sys.exit(1)
