#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대용량 텍스트 요약 성능 테스트
LexRank와 LSA 알고리즘을 사용하여 1GB 텍스트 요약
"""

import time
import sys
import os
from datetime import datetime

# sumy 경로 추가
sys.path.insert(0, os.path.abspath('..'))

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


def format_time(seconds):
    """시간을 읽기 쉬운 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.2f}초"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}분 {secs:.2f}초"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}시간 {int(minutes)}분 {secs:.2f}초"


def print_timestamp(label):
    """타임스탬프 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {label}")


def test_summarization(input_file, language="korean", sentences_count=10):
    """텍스트 요약 성능 테스트"""

    print("=" * 80)
    print("대용량 텍스트 요약 성능 테스트")
    print("=" * 80)
    print(f"입력 파일: {input_file}")
    print(f"언어: {language}")
    print(f"요약 문장 수: {sentences_count}")
    print("=" * 80)

    # 파일 크기 확인
    file_size = os.path.getsize(input_file)
    file_size_mb = file_size / 1024 / 1024
    file_size_gb = file_size / 1024 / 1024 / 1024

    print(f"\n파일 크기: {file_size:,} bytes ({file_size_mb:.2f} MB / {file_size_gb:.3f} GB)")

    # 1. 파일 읽기
    print("\n" + "-" * 80)
    print("1단계: 파일 읽기")
    print("-" * 80)
    print_timestamp("시작: 파일 읽기")

    start_time = time.time()
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    read_time = time.time() - start_time

    print_timestamp("완료: 파일 읽기")
    print(f"소요 시간: {format_time(read_time)}")
    print(f"텍스트 길이: {len(text):,} 문자")

    # 2. 파서 생성 및 토큰화
    print("\n" + "-" * 80)
    print("2단계: 파서 생성 및 토큰화")
    print("-" * 80)
    print_timestamp("시작: 토큰화")

    start_time = time.time()
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    tokenize_time = time.time() - start_time

    print_timestamp("완료: 토큰화")
    print(f"소요 시간: {format_time(tokenize_time)}")
    print(f"문장 수: {len(parser.document.sentences):,}")

    stemmer = Stemmer(language)

    # 3. LexRank 알고리즘 테스트
    print("\n" + "=" * 80)
    print("알고리즘 1: LexRank")
    print("=" * 80)
    print("설명: 그래프 기반 알고리즘, 문장 간 유사도를 사용")
    print("-" * 80)

    print_timestamp("시작: LexRank 요약")
    start_time = time.time()

    summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)

    try:
        summary = list(summarizer(parser.document, sentences_count))
        lexrank_time = time.time() - start_time

        print_timestamp("완료: LexRank 요약")
        print(f"소요 시간: {format_time(lexrank_time)}")
        print(f"요약 문장 수: {len(summary)}")

        print("\n요약 결과 (처음 3문장):")
        for i, sentence in enumerate(summary[:3], 1):
            print(f"  {i}. {str(sentence)[:100]}...")

    except Exception as e:
        print_timestamp("오류: LexRank 요약")
        print(f"에러: {e}")
        lexrank_time = None

    # 4. LSA 알고리즘 테스트
    print("\n" + "=" * 80)
    print("알고리즘 2: LSA (Latent Semantic Analysis)")
    print("=" * 80)
    print("설명: 행렬 분해를 사용하여 주제 추출")
    print("-" * 80)

    print_timestamp("시작: LSA 요약")
    start_time = time.time()

    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)

    try:
        summary = list(summarizer(parser.document, sentences_count))
        lsa_time = time.time() - start_time

        print_timestamp("완료: LSA 요약")
        print(f"소요 시간: {format_time(lsa_time)}")
        print(f"요약 문장 수: {len(summary)}")

        print("\n요약 결과 (처음 3문장):")
        for i, sentence in enumerate(summary[:3], 1):
            print(f"  {i}. {str(sentence)[:100]}...")

    except Exception as e:
        print_timestamp("오류: LSA 요약")
        print(f"에러: {e}")
        lsa_time = None

    # 5. 결과 요약
    print("\n" + "=" * 80)
    print("성능 테스트 결과 요약")
    print("=" * 80)

    total_time = read_time + tokenize_time
    if lexrank_time:
        total_time += lexrank_time
    if lsa_time:
        total_time += lsa_time

    print(f"\n파일 정보:")
    print(f"  - 크기: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
    print(f"  - 문자 수: {len(text):,}")
    print(f"  - 문장 수: {len(parser.document.sentences):,}")

    print(f"\n처리 시간:")
    print(f"  1. 파일 읽기:     {format_time(read_time)}")
    print(f"  2. 토큰화:        {format_time(tokenize_time)}")
    if lexrank_time:
        print(f"  3. LexRank 요약:  {format_time(lexrank_time)}")
    if lsa_time:
        print(f"  4. LSA 요약:      {format_time(lsa_time)}")
    print(f"  ───────────────────────────────")
    print(f"  총 소요 시간:     {format_time(total_time)}")

    print(f"\n처리 속도:")
    print(f"  - 읽기 속도: {file_size_mb / read_time:.2f} MB/s")
    print(f"  - 토큰화 속도: {len(text) / tokenize_time / 1000:.2f}K 문자/초")
    if lexrank_time:
        print(f"  - LexRank 속도: {len(parser.document.sentences) / lexrank_time:.2f} 문장/초")
    if lsa_time:
        print(f"  - LSA 속도: {len(parser.document.sentences) / lsa_time:.2f} 문장/초")

    print("\n" + "=" * 80)
    print_timestamp("테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    # 1GB 텍스트 파일 테스트
    test_summarization(
        input_file="large_text_1gb.txt",
        language="english",  # 한국어 텍스트이지만 영어 토크나이저 사용 (kiwipiepy 미설치)
        sentences_count=10
    )
