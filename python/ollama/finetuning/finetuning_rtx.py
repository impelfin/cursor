## finetuning.py

import os
import json
import torch
import sys
import subprocess
import logging
from datetime import datetime # 출력 디렉터리에 타임스탬프를 추가하기 위해 추가

# Hugging Face 관련 라이브러리 (pip install 필수!)
from datasets import Dataset # <<< 문제의 Dataset 임포트
from transformers import AutoModelForCausalLM, AutoTokenizer # <<< 문제의 AutoModelForCausalLM 임포트
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

# 로깅 설정: 스크립트 진행 상황을 명확하게 확인하기 위해 INFO 레벨로 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

logger.info("모든 라이브러리 임포트 및 로깅 설정 완료.") # <<< 이 줄이 출력되는지 확인!


# --- 1. 설정 변수 ---
# 파인튜닝할 원본 기본 모델의 로컬 경로.
# 이 경로에 Hugging Face 형식의 모델 파일들(config.json, model.safetensors 등)이 있어야 합니다.
base_model_local_path = "./llama-3.2-1b"

# 학습 데이터셋 JSON 파일 경로. 1시간 이내 완료를 위해 샘플 수를 대폭 줄여야 합니다!
# "sft.json" 파일의 샘플 수가 많다면, 줄여서 "sft_short.json"으로 만들고 그 경로를 사용하세요.
sft_json_path = "./sft.json" # <<< "sft_short.json" 대신 "sft.json"으로 되어있으니 확인 필요.

# 파인튜닝 결과 및 병합 모델이 저장될 디렉터리.
# 타임스탬프를 사용하지 않고 고정된 폴더에 저장됩니다. (기존 파일 덮어쓰기 가능성 있음!)
output_dir = "./finetuned-llama-3.2-1b" # <<< 타임스탬프 대신 고정된 폴더명 사용.
os.makedirs(output_dir, exist_ok=True) # 출력 디렉터리 생성 (존재하면 오류 없음)

# GGUF 변환 결과 파일 이름 및 경로.
gguf_output_name = f"{os.path.basename(base_model_local_path).lower()}-finetuned.gguf"
gguf_output_path = os.path.join(output_dir, gguf_output_name)

# llama.cpp 저장소의 루트 경로. convert_hf_to_gguf.py 스크립트가 이 경로 아래에 있어야 합니다.
llama_cpp_path = "llama.cpp"

logger.info("설정 변수 로드 완료.") # <<< 이 줄이 출력되는지 확인!


# --- 2. 모델 및 토크나이저 로드 ---
logger.info(f"모델 로드 중: {base_model_local_path}...")

# --- NVIDIA CUDA GPU 사용 가능 여부 확인 로직 (Windows용) ---
if torch.cuda.is_available():
    device = "cuda"
    logger.info(f"CUDA (NVIDIA GPU)가 사용 가능합니다. 디바이스: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM). 이를 사용하여 학습을 가속합니다.")
else:
    device = "cpu"
    logger.warning("CUDA (NVIDIA GPU)를 사용할 수 없습니다. CPU를 사용합니다. 학습 속도가 매우 느릴 수 있습니다.")
    logger.error("GPU를 사용할 수 없으므로 파인튜닝을 계속할 수 없습니다. CUDA 설치 및 드라이버를 확인하세요.")
    exit(1) # GPU 없으면 바로 종료

try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_local_path,
        # quantization_config=bnb_config,
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

    logger.info("모델과 토크나이저 로드 성공.") # <<< 이 줄이 출력되는지 확인!

except Exception as e:
    logger.error(f"\n--- 오류: 모델 또는 토크나이저 로드 중 오류 발생 ---")
    logger.error(f"원인: {e}")
    logger.error(f"'{base_model_local_path}' 경로에 모델 파일이 완전히 존재하는지, PyTorch/Transformers 설치가 올바른지 확인하세요.")
    logger.error("추가 팁: Windows의 모든 애플리케이션을 종료하여 GPU VRAM을 최대한 확보하세요.")
    logger.error("2. 모델 파일이 손상되었거나 불완전하게 다운로드되었을 수 있으니, 다시 다운로드해보세요.") # 줄바꿈된 듯, 원래는 1줄
    logger.error("3. `base_model_local_path`를 `EleutherAI/gpt-neo-125M` 같은 훨씬 작은 모델로 변경하여 스크립트가 작동하는지 테스트해 보세요.")
    exit(1)


# --- 3. 데이터셋 로드 및 전처리 ---
logger.info(f"데이터셋 로드 중: {sft_json_path}...")
try:
    with open(sft_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list(raw_data)
    logger.info(f"데이터셋 로드 성공. 총 {len(dataset)}개의 샘플을 학습합니다.")
except FileNotFoundError:
    logger.error(f"오류: {sft_json_path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit(1)
except json.JSONDecodeError:
    logger.error(f"오류: {sft_json_path} 파일이 유효한 JSON 형식이 아닙니다. JSON 포맷을 확인해주세요.")
    exit(1)
except Exception as e:
    logger.error(f"데이터셋 로드 중 오류 발생: {e}")
    exit(1)


# --- 4. LoRA 설정 ---
# LoRA는 적은 수의 파라미터만 학습하여 메모리 사용량을 최소화하면서 파인튜닝을 가능하게 한다.
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
logger.info("LoRA 어댑터가 모델에 성공적으로 적용되었습니다.")


# --- 6. 학습 인자 설정 ---
# 1시간 이내 완료를 위한 학습 파라미터 설정 (테스트용)
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
    fp16=True, # NVIDIA GPU의 FP16 가속 활용 (메모리 절약 및 속도 향상)
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    disable_tqdm=False,
    max_seq_length=128, # 시퀀스 길이를 128 토큰으로 제한 (시간 단축 핵심)
    dataset_text_field="text",
    packing=False,
)


# --- 7. SFTTrainer 설정 및 학습 시작 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_training_args,
    # tokenizer=tokenizer, # 일반적으로 Trainer가 자동으로 토크나이저를 사용
)

logger.info("\n--- 파인튜닝 시작 ---")
try:
    trainer.train() # <<< 학습을 진행합니다.
    logger.info("\n--- 파인튜닝 완료 ---")
except Exception as e:
    logger.error(f"\n--- 오류: 파인튜닝 중 오류 발생 ---")
    logger.error(f"원인: {e}")
    logger.error("메모리 부족, 학습 인자 설정 오류 등을 확인하세요. `gradient_accumulation_steps`를 늘리거나 `max_seq_length`를 줄여보세요.")
    logger.error("\n--- NVIDIA GPU 관련 문제 해결 팁 ---")
    logger.error("1. `per_device_train_batch_size`를 줄여보세요 (최소 1).")
    logger.error("2. `max_seq_length`를 줄여보세요 (예: 64 또는 32).")
    logger.error("3. `torch_dtype`을 `torch.bfloat16` 대신 `torch.float16`으로 유지하고, `fp16=True`로 설정했는지 확인하세요.")
    logger.error("4. `bitsandbytes` 라이브러리가 올바르게 설치되었는지 확인하세요.")
    exit(1)


# --- 8. 파인튜닝된 LoRA 어댑터와 토크나이저 저장 ---
# 학습이 완료되면 trainer가 자동으로 output_dir에 저장하지만, 명시적으로 다시 저장하는 것은 안전합니다.
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"\n파인튜닝된 LoRA 어댑터와 토크나이저가 '{output_dir}'에 저장되었습니다.")


# --- 9. LoRA 어댑터를 기본 모델에 병합 및 전체 모델 저장 (GGUF 변환을 위해) ---
logger.info("\nLoRA 어댑터를 기본 모델에 병합 중 (GGUF 변환 준비)...")

try:
    # 병합할 때도 기본 모델을 로드할 때 `torch_dtype`을 `float16`으로 일치시킴.
    # GPU VRAM이 부족할 경우 CPU RAM을 적극적으로 활용하도록 offload_buffers=True 추가.
    base_model_full = AutoModelForCausalLM.from_pretrained(
        base_model_local_path,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        offload_buffers=True # VRAM 부족 오류 방지를 위해 이 줄 추가
    )

    # 파인튜닝된 LoRA 어댑터를 기본 모델에 로드
    # 이전에 저장된 LoRA 어댑터가 output_dir에 있다고 가정합니다.
    model_to_merge = PeftModel.from_pretrained(base_model_full, output_dir)
    merged_model = model_to_merge.merge_and_unload() # LoRA 가중치를 기본 모델에 병합하여 하나의 완전한 모델 생성

    merged_model_save_path = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_model_save_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_save_path) # 병합된 모델과 토크나이저 함께 저장

    logger.info(f"\n병합된 모델이 '{merged_model_save_path}'에 저장되었습니다. 이제 GGUF로 변환할 수 있습니다.")

except Exception as e:
    logger.error(f"\n--- 오류: LoRA 어댑터 병합 중 오류 발생 ---")
    logger.error(f"원인: {e}")
    logger.error(f"'{base_model_local_path}' 경로의 기본 모델 파일이 온전한지, 그리고 Windows에 충분한 메모리가 있는지 확인하세요.")
    exit(1)


# --- 10. GGUF 변환 (llama.cpp의 convert_hf_to_gguf.py 사용) ---
convert_py_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")

if not os.path.exists(convert_py_path):
    logger.warning(f"\n경고: '{convert_py_path}'를 찾을 수 없습니다.")
    logger.warning("GGUF 변환을 위해서는 llama.cpp 레포지토리와 해당 스크립트가 필요합니다. 위의 '스크립트 실행 전 필수 확인 사항'을 다시 확인하세요.")
    exit(1)
else:
    logger.info(f"\nGGUF 변환 시작: {gguf_output_path}...")
    # GGUF 변환 시에도 메모리 절약을 위해 `--outtype F16`을 권장.
    # F32는 파일 크기가 매우 크고 추론 시 더 많은 메모리를 요구함.
    # `subprocess.run`에서 가상 환경의 Python 인터프리터를 명시적으로 사용.
    python_executable = os.path.join(os.path.dirname(sys.executable), "python.exe")

    convert_command = [
        python_executable, # 'python' 대신 가상 환경의 Python 실행 파일 경로 사용
        convert_py_path,
        merged_model_save_path, # 병합된 모델 경로 사용
        "--outfile", gguf_output_path,
        "--outtype", "f16" # 'F16' 대신 'f16'으로 변경!
    ]
    logger.info(f"실행 명령: {' '.join(convert_command)}")

    try:
        process = subprocess.run(convert_command, check=True, capture_output=True, text=True)
        logger.info("GGUF 변환 출력:")
        logger.info(process.stdout)
        if process.stderr:
            logger.warning("GGUF 변환 경고/오류 (있는 경우):")
            logger.warning(process.stderr)

    except subprocess.CalledProcessError as e:
        logger.error(f"\n--- 오류: GGUF 변환 중 오류 발생 ---")
        logger.error(f"명령어: {' '.join(e.cmd)}")
        logger.error(f"반환 코드: {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        logger.error("llama.cpp가 빌드되었고, 변환 스크립트 경로가 올바른지 확인하세요.")
        exit(1)
    except FileNotFoundError:
        logger.error(f"\n--- 오류: 'python' 명령어를 찾을 수 없습니다. Python이 PATH에 추가되었는지 확인하세요. ---")
        logger.error(f"현재 Python 실행 경로: {sys.executable}")
        exit(1)
    except Exception as e:
        logger.error(f"\n--- 예상치 못한 오류: {e} ---")
        exit(1)

    if os.path.exists(gguf_output_path):
        logger.info(f"\n--- 성공: 파인튜닝된 GGUF 모델이 '{gguf_output_path}'에 생성되었습니다. ---")
        logger.info("이제 이 GGUF 파일을 사용하여 Ollama에 모델을 등록하고 실행할 수 있습니다.")
        logger.info("\n--- 다음 단계 ---")
        logger.info(f"1. Ollama bash 스크립트 또는 Modelfile에서 `GGUF_MODEL_PATH`가 '{gguf_output_path}'를 정확히 가리키는지 확인하세요.")
        logger.info("2. Ollama bash 스크립트를 다시 실행하거나 `ollama create` 명령어를 사용하여 모델을 등록하고 실행하세요.")
    else:
        logger.error("\n— 오류: GGUF 모델 생성에 실패했습니다. llama.cpp 변환 과정을 다시 확인해주세요. —")
        exit(1)
