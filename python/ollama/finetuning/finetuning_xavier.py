import os
import json
import torch
import sys
import subprocess
import logging
from datetime import datetime

# Hugging Face 관련 라이브러리 (pip install 필수!)
# Jetson에서 설치: pip install transformers peft datasets trl accelerate
# 주의: bitsandbytes는 Jetson에서 설치가 매우 어렵거나 불가능합니다. 이 스크립트에서는 사용하지 않습니다.
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

# 로깅 설정: 스크립트 진행 상황을 명확하게 확인하기 위해 INFO 레벨로 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

logger.info("모든 라이브러리 임포트 및 로깅 설정 완료.")


# --- 1. 설정 변수 ---
# 파인튜닝할 원본 기본 모델의 로컬 경로.
# 이 경로에 Hugging Face 형식의 모델 파일들(config.json, model.safetensors 등)이 있어야 합니다.
# 예: Llama-3.2-1B 모델 파일을 이 경로에 다운로드해 놓으세요.
base_model_local_path = "./llama-3.2-1b"

# 학습 데이터셋 JSON 파일 경로.
# "sft.json" 파일의 샘플 수가 많다면, Jetson의 제한된 리소스를 고려하여
# 샘플 수를 대폭 줄인 "sft_short.json" 같은 파일을 사용하는 것이 좋습니다.
sft_json_path = "./sft.json"

# 파인튜닝 결과 및 병합 모델이 저장될 디렉터리.
# (기존 파일 덮어쓰기 가능성 있음!)
output_dir = "./finetuned-llama-3.2-1b"
os.makedirs(output_dir, exist_ok=True) # 출력 디렉터리 생성 (존재하면 오류 없음)

# GGUF 변환 결과 파일 이름 및 경로.
gguf_output_name = f"{os.path.basename(base_model_local_path).lower()}-finetuned.gguf"
gguf_output_path = os.path.join(output_dir, gguf_output_name)

# llama.cpp 저장소의 루트 경로. convert_hf_to_gguf.py 스크립트가 이 경로 아래에 있어야 합니다.
# Jetson에서 llama.cpp를 클론하고 make 명령어로 빌드했는지 확인하세요.
llama_cpp_path = "llama.cpp"

logger.info("설정 변수 로드 완료.")


# --- 2. 모델 및 토크나이저 로드 ---
logger.info(f"모델 로드 중: {base_model_local_path}...")

# --- Jetson NVIDIA CUDA GPU 사용 가능 여부 확인 ---
if torch.cuda.is_available():
    device = "cuda"
    # Jetson은 일반적으로 하나의 GPU (0번)를 가집니다.
    logger.info(f"CUDA (NVIDIA GPU)가 사용 가능합니다. 디바이스: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM). 이를 사용하여 학습을 가속합니다.")
else:
    device = "cpu"
    logger.error("CUDA (NVIDIA GPU)를 사용할 수 없습니다. CPU를 사용합니다. 학습 속도가 매우 느려 파인튜닝이 사실상 불가능합니다.")
    logger.error("JetPack 설치가 올바른지, 그리고 PyTorch가 Jetson용으로 올바르게 설치되었는지 확인하세요.")
    sys.exit(1) # GPU 없으면 바로 종료

try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_local_path,
        # Jetson에서 bitsandbytes 사용은 어렵습니다. 이 부분은 활성화하지 마세요.
        # quantization_config=bnb_config,
        device_map="auto", # 자동으로 사용 가능한 디바이스에 모델 할당
        torch_dtype=torch.float16, # Jetson의 메모리 효율을 위해 float16 사용 (BF16 대신)
        low_cpu_mem_usage=True, # CPU 메모리 사용을 줄여 GPU로 오프로드 시도
        trust_remote_code=True, # 커스텀 코드 실행 허용 (필요시)
        local_files_only=True # 로컬 파일만 사용 (네트워크 다운로드 시도 안함)
    )
    model.config.use_cache = False # 학습 중 캐싱 비활성화

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_local_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # 패딩 토큰 설정
    tokenizer.padding_side = "right" # 패딩 위치 설정

    logger.info("모델과 토크나이저 로드 성공.")

except Exception as e:
    logger.error(f"\n--- 오류: 모델 또는 토크나이저 로드 중 오류 발생 ---")
    logger.error(f"원인: {e}")
    logger.error(f"'{base_model_local_path}' 경로에 모델 파일이 완전히 존재하는지 확인하세요.")
    logger.error("Jetson의 모든 애플리케이션을 종료하여 GPU VRAM을 최대한 확보하세요.")
    logger.error("모델 파일이 손상되었거나 불완전하게 다운로드되었을 수 있으니, 다시 다운로드해보세요.")
    logger.error("`base_model_local_path`를 `EleutherAI/gpt-neo-125M` 같은 훨씬 작은 모델로 변경하여 스크립트가 작동하는지 테스트해 보세요.")
    sys.exit(1)


# --- 3. 데이터셋 로드 및 전처리 ---
logger.info(f"데이터셋 로드 중: {sft_json_path}...")
try:
    with open(sft_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 데이터셋의 각 샘플이 텍스트 필드를 가지도록 변환 (sft.json의 형식에 따라 이 함수를 수정해야 합니다!)
    # 예시: [{"instruction": "질문", "output": "답변"}] 형태의 JSON을
    # {"text": "### Instruction:\n질문\n### Output:\n답변"} 형태로 변환
    def format_data_for_sft(example):
        # sft.json 파일의 실제 키에 맞춰 수정하세요.
        # 아래는 가장 일반적인 instruct-tuning 형식입니다.
        if "instruction" in example and "output" in example:
            text = f"### Instruction:\n{example['instruction']}\n### Output:\n{example['output']}"
        elif "prompt" in example and "completion" in example:
            text = f"### Prompt:\n{example['prompt']}\n### Completion:\n{example['completion']}"
        elif "text" in example: # 이미 통합된 text 필드인 경우
            text = example["text"]
        else:
            logger.error(f"데이터셋 샘플에 'instruction'/'output' 또는 'prompt'/'completion' 또는 'text' 필드가 없습니다: {example}")
            sys.exit(1)
        return {"text": text}

    # Dataset 객체로 변환하고 포맷팅 적용
    dataset = Dataset.from_list(raw_data).map(format_data_for_sft, remove_columns=list(raw_data[0].keys()))

    logger.info(f"데이터셋 로드 및 포맷팅 성공. 총 {len(dataset)}개의 샘플을 학습합니다.")
except FileNotFoundError:
    logger.error(f"오류: {sft_json_path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    sys.exit(1)
except json.JSONDecodeError:
    logger.error(f"오류: {sft_json_path} 파일이 유효한 JSON 형식이 아닙니다. JSON 포맷을 확인해주세요.")
    sys.exit(1)
except Exception as e:
    logger.error(f"데이터셋 로드 또는 전처리 중 오류 발생: {e}")
    sys.exit(1)


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
# Jetson NX의 메모리 및 컴퓨팅 제한을 고려한 학습 파라미터 설정
sft_training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1, # 1 에포크로 학습 시간 단축 (테스트용. 실제 학습 시에는 3~5 에포크 고려)
    per_device_train_batch_size=1, # Jetson의 GPU 메모리 제약으로 인해 배치 크기를 1로 설정
    gradient_accumulation_steps=4, # 작은 배치 크기를 보완하기 위한 그래디언트 누적 단계 (VRAM 절약)
    optim="adamw_torch", # AdamW optimizer 사용
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_steps=50,
    logging_steps=10,
    push_to_hub=False,
    report_to="none",
    fp16=True, # NVIDIA GPU의 FP16 가속 활용 (메모리 절약 및 속도 향상)
    bf16=False, # Jetson은 보통 bf16을 지원하지 않습니다. (Ampere 아키텍처 이상에서 지원)
    max_grad_norm=0.3, # 그래디언트 클리핑
    warmup_ratio=0.03,
    group_by_length=True,
    disable_tqdm=False,
    max_seq_length=128, # 시퀀스 길이를 128 토큰으로 제한 (메모리 및 시간 단축 핵심)
    dataset_text_field="text",
    packing=False, # Packing은 데이터셋 길이가 긴 경우 유용하지만, 여기서는 max_seq_length로 제한
)


# --- 7. SFTTrainer 설정 및 학습 시작 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_training_args,
    tokenizer=tokenizer, # SFTTrainer에 토크나이저 명시적 전달 권장
)

logger.info("\n--- 파인튜닝 시작 ---")
try:
    trainer.train()
    logger.info("\n--- 파인튜닝 완료 ---")
except Exception as e:
    logger.error(f"\n--- 오류: 파인튜닝 중 오류 발생 ---")
    logger.error(f"원인: {e}")
    logger.error("메모리 부족, 학습 인자 설정 오류 등을 확인하세요.")
    logger.error("팁:")
    logger.error("1. `per_device_train_batch_size`를 줄여보세요 (최소 1).")
    logger.error("2. `gradient_accumulation_steps`를 늘려보세요.")
    logger.error("3. `max_seq_length`를 줄여보세요 (예: 64 또는 32).")
    logger.error("4. Jetson의 모든 불필요한 앱을 종료하여 GPU VRAM을 확보하세요.")
    sys.exit(1)


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
        torch_dtype=torch.float16, # float16으로 일관성 유지
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        offload_buffers=True # VRAM 부족 오류 방지를 위해 이 줄 추가
    )

    # 파인튜닝된 LoRA 어댑터를 기본 모델에 로드
    model_to_merge = PeftModel.from_pretrained(base_model_full, output_dir)
    merged_model = model_to_merge.merge_and_unload() # LoRA 가중치를 기본 모델에 병합하여 하나의 완전한 모델 생성

    merged_model_save_path = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_model_save_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_save_path) # 병합된 모델과 토크나이저 함께 저장

    logger.info(f"\n병합된 모델이 '{merged_model_save_path}'에 저장되었습니다. 이제 GGUF로 변환할 수 있습니다.")

except Exception as e:
    logger.error(f"\n--- 오류: LoRA 어댑터 병합 중 오류 발생 ---")
    logger.error(f"원인: {e}")
    logger.error(f"'{base_model_local_path}' 경로의 기본 모델 파일이 온전한지, 그리고 Jetson에 충분한 메모리가 있는지 확인하세요.")
    sys.exit(1)


# --- 10. GGUF 변환 (llama.cpp의 convert_hf_to_gguf.py 사용) ---
convert_py_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")

if not os.path.exists(convert_py_path):
    logger.warning(f"\n경고: '{convert_py_path}'를 찾을 수 없습니다.")
    logger.warning("GGUF 변환을 위해서는 llama.cpp 레포지토리와 해당 스크립트가 필요합니다. Jetson에서 llama.cpp를 클론하고 빌드했는지 확인하세요.")
    sys.exit(1)
else:
    logger.info(f"\nGGUF 변환 시작: {gguf_output_path}...")
    # `subprocess.run`에서 현재 환경의 Python 인터프리터를 명시적으로 사용.
    # Jetson은 Linux 기반이므로 'python.exe' 대신 'python' 또는 'python3'가 적합합니다.
    # sys.executable은 현재 실행 중인 Python 인터프리터의 전체 경로를 반환합니다.
    python_executable = sys.executable

    convert_command = [
        python_executable, # 현재 Python 실행 파일 경로 사용
        convert_py_path,
        merged_model_save_path, # 병합된 모델 경로 사용
        "--outfile", gguf_output_path,
        "--outtype", "f16" # Jetson의 메모리 제한을 고려하여 'f16' 또는 'q4_0', 'q5_k_m' 권장
    ]
    logger.info(f"실행 명령: {' '.join(convert_command)}")

    try:
        # GGUF 변환은 시간이 걸릴 수 있으므로, 출력 스트림을 실시간으로 확인하는 것이 좋습니다.
        # check=True로 설정하여 오류 발생 시 예외를 발생시키고, capture_output=False로 실시간 출력 허용.
        process = subprocess.run(convert_command, check=True)
        # stderr와 stdout은 capture_output=False일 때 직접 콘솔로 나옵니다.
        # 따라서 여기서 process.stdout, process.stderr를 출력할 필요는 없습니다.

    except subprocess.CalledProcessError as e:
        logger.error(f"\n--- 오류: GGUF 변환 중 오류 발생 ---")
        logger.error(f"명령어: {' '.join(e.cmd)}")
        logger.error(f"반환 코드: {e.returncode}")
        # 오류 발생 시에만 stderr를 다시 확인 (capture_output=True였다면)
        # logger.error(f"stdout: {e.stdout}")
        # logger.error(f"stderr: {e.stderr}")
        logger.error("llama.cpp가 빌드되었고, 변환 스크립트 경로가 올바른지 확인하세요.")
        logger.error("또한, Jetson의 RAM이 충분한지 확인하세요. F16 변환 시에도 많은 RAM이 필요할 수 있습니다.")
        logger.error("만약 RAM 부족이 의심되면, `--outtype q4_0` 또는 `--outtype q5_k_m`으로 변경하여 시도해 보세요.")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"\n--- 오류: 'python' 명령어를 찾을 수 없습니다. Python이 PATH에 추가되었거나, `sys.executable` 경로가 올바른지 확인하세요. ---")
        logger.error(f"현재 Python 실행 경로: {sys.executable}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n--- 예상치 못한 오류: {e} ---")
        sys.exit(1)

    if os.path.exists(gguf_output_path):
        logger.info(f"\n--- 성공: 파인튜닝된 GGUF 모델이 '{gguf_output_path}'에 생성되었습니다. ---")
        logger.info("이제 이 GGUF 파일을 사용하여 Ollama에 모델을 등록하고 실행할 수 있습니다.")
        logger.info("\n--- 다음 단계 ---")
        logger.info(f"1. Modelfile을 생성하여 `FROM {gguf_output_path}`를 포함하고, Ollama에 모델을 등록하세요.")
        logger.info(f"   예시 Modelfile 내용:")
        logger.info(f"   ```")
        logger.info(f"   FROM {gguf_output_path}")
        logger.info(f"   PARAMETER stop \"<|eot_id|>\"")
        logger.info(f"   PARAMETER stop \"<|start_header_id|>\"")
        logger.info(f"   PARAMETER stop \"<|end_header_id|>\"")
        logger.info(f"   # SYSTEM \"You are a helpful AI assistant trained on specific tasks.\"")
        logger.info(f"   ```")
        logger.info(f"2. 터미널에서 다음 명령을 실행하여 Ollama에 모델을 등록합니다:")
        logger.info(f"   ollama create finetuned-llama-3.2-1b -f ./Modelfile")
        logger.info(f"3. 등록된 모델을 실행합니다:")
        logger.info(f"   ollama run finetuned-llama-3.2-1b")
    else:
        logger.error("\n--- 오류: GGUF 모델 생성에 실패했습니다. llama.cpp 변환 과정을 다시 확인해주세요. ---")
        sys.exit(1)
