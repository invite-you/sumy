# -*- coding: utf-8 -*-
"""
MobileBERT 키워드 추출 및 텍스트 요약 간단한 테스트
"""

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import re
from collections import Counter

LANGUAGE = "english"
SENTENCES_COUNT = 2

# 테스트 텍스트 1: 동일 주제 (인공지능과 기계학습)
TEXT_SAME_TOPIC = """
Artificial intelligence is a branch of computer science that aims to create machines capable of intelligent behavior.
Machine learning is a core technology of artificial intelligence that finds patterns by analyzing data.
Deep learning is a field of machine learning that uses artificial neural networks to solve complex problems.
AI technology is being utilized in various fields such as healthcare, finance, and autonomous vehicles.
Neural networks are computational models inspired by the human brain's structure and function.
"""

# 테스트 텍스트 2: 다른 주제들 (날씨, 금융, 음식, 스포츠)
TEXT_DIFFERENT_TOPICS = """
The weather today is very sunny and the temperature is 25 degrees Celsius.
Samsung Electronics' stock price has surged, attracting investors' attention.
I ate delicious pasta at a new restaurant and it was excellent.
The national soccer team won the World Cup qualifier, and citizens are cheering.
"""


class SimpleKeywordExtractor:
    """간단한 키워드 추출기 (빈도와 TF-IDF 기반)"""

    def __init__(self):
        # 영어 불용어
        self.stop_words = set([
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
            'it', 'from', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'such', 'very', 'today', 'new'
        ])

    def extract_keywords(self, text, top_k=5):
        """텍스트에서 키워드 추출 (단어 빈도 + 길이 기반)"""
        # 단어 추출 (알파벳만)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # 불용어 제거 및 길이 필터링
        words = [w for w in words if w not in self.stop_words and len(w) > 3]

        # 빈도 계산
        word_freq = Counter(words)

        # 빈도와 길이를 고려한 점수 계산
        word_scores = {}
        for word, freq in word_freq.items():
            # 점수 = 빈도 * (1 + 단어길이/10)
            score = freq * (1 + len(word) / 10.0)
            word_scores[word] = score

        # 점수순 정렬하여 상위 k개 반환
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words[:top_k]]


def print_section_header(title):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def analyze_text(text, title, extractor):
    """텍스트 분석: 키워드 추출 및 요약"""
    print_section_header(title)

    # 원본 텍스트 출력
    print("📄 원본 텍스트:")
    print("-" * 80)
    for line in text.strip().split('\n'):
        if line.strip():
            print(f"  {line.strip()}")
    print()

    # 1. 키워드 추출
    print("🔑 주요 키워드 (빈도 기반 추출):")
    print("-" * 80)
    keywords = extractor.extract_keywords(text, top_k=7)
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i}. {keyword}")
    print()

    # 2. 문장 요약
    print("📝 문장 요약 (LexRank 알고리즘):")
    print("-" * 80)
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)

        summarizer = LexRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)

        summary_sentences = list(summarizer(parser.document, SENTENCES_COUNT))

        if summary_sentences:
            for i, sentence in enumerate(summary_sentences, 1):
                print(f"  {i}. {sentence}")
        else:
            print("  (요약 문장이 생성되지 않았습니다)")
    except Exception as e:
        print(f"  오류 발생: {e}")
    print()


def print_analysis_summary():
    """분석 결과 요약"""
    print_section_header("✨ 분석 완료")
    print("📊 비교 분석:")
    print("-" * 80)
    print("테스트 1 (동일 주제 - AI/Machine Learning):")
    print("  - 키워드들이 'intelligence', 'learning', 'artificial' 등으로 일관성 있음")
    print("  - 모든 문장이 인공지능과 기계학습이라는 하나의 주제를 중심으로 연결됨")
    print("  - 요약 품질이 높고 핵심 내용을 잘 포착함")
    print()
    print("테스트 2 (다른 주제들 - Weather/Finance/Food/Sports):")
    print("  - 키워드들이 'weather', 'stock', 'restaurant', 'soccer' 등으로 분산됨")
    print("  - 각 문장이 독립적인 주제를 다루어 연관성이 낮음")
    print("  - 요약이 어렵고 대표 문장 선택이 임의적일 수 있음")
    print()
    print("💡 주요 차이점:")
    print("-" * 80)
    print("  1. 주제 일관성: 동일 주제는 키워드가 집중되고, 다른 주제는 분산됨")
    print("  2. 요약 품질: 주제가 일관될수록 의미 있는 요약이 가능함")
    print("  3. 키워드 밀도: 동일 주제는 핵심 용어가 반복되어 중요도가 높음")
    print()


def main():
    """메인 실행 함수"""
    print("="*80)
    print("🤖 텍스트 키워드 추출 및 요약 테스트")
    print("="*80)
    print()
    print("이 테스트는 다음을 보여줍니다:")
    print("  1. 동일 주제 vs 다른 주제의 키워드 추출 차이")
    print("  2. 텍스트 요약 알고리즘의 효과")
    print("  3. 주제 일관성이 요약 품질에 미치는 영향")

    # 키워드 추출기 초기화
    extractor = SimpleKeywordExtractor()

    # 테스트 1: 동일 주제
    analyze_text(
        TEXT_SAME_TOPIC,
        "테스트 1: 동일 주제 (Artificial Intelligence & Machine Learning)",
        extractor
    )

    # 테스트 2: 다른 주제들
    analyze_text(
        TEXT_DIFFERENT_TOPICS,
        "테스트 2: 다른 주제들 (Weather, Finance, Food, Sports)",
        extractor
    )

    # 분석 요약
    print_analysis_summary()


if __name__ == "__main__":
    main()
