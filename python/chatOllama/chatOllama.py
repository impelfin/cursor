from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote.messages import stream_response

# API 키 정보 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("ChatOllama")

# Ollama 모델 지정
llm = ChatOllama(
    model="exaone",
    temperature=0,
)

# 프롬프트 정의
prompt = ChatPromptTemplate.from_template("{topic} 에 대하여 간략히 설명해 줘.")

# 체인 생성
chain = prompt | llm | StrOutputParser()

# 스트림 출력
answer = chain.stream({"topic": "deep learning"})
stream_response(answer)