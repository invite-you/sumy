# -*- coding: utf-8 -*-
"""
한국어 예제 1: 동일한 주제의 긴 내용 요약
주제: 인공지능과 머신러닝의 발전
"""

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# 모든 요약 알고리즘 import
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.reduction import ReductionSummarizer


LANGUAGE = "korean"
SENTENCES_COUNT = 3

# 동일한 주제: 인공지능과 머신러닝에 대한 긴 텍스트
TEXT = """
인공지능은 컴퓨터 과학의 한 분야로서 기계가 인간과 유사한 지능을 가지도록 만드는 기술입니다.
인공지능의 역사는 1950년대로 거슬러 올라가며, 앨런 튜링이 기계가 사고할 수 있는지에 대한 질문을 제기하면서 시작되었습니다.
현대 인공지능은 크게 약한 인공지능과 강한 인공지능으로 구분됩니다.
약한 인공지능은 특정 작업을 수행하도록 설계된 시스템으로, 현재 우리가 일상에서 사용하는 대부분의 AI가 여기에 속합니다.
강한 인공지능은 인간 수준의 일반 지능을 가진 시스템을 의미하며, 아직 실현되지 않았습니다.

머신러닝은 인공지능의 핵심 기술 중 하나로, 데이터로부터 패턴을 학습하여 예측이나 결정을 내리는 방법입니다.
전통적인 프로그래밍이 명시적인 규칙을 코딩하는 것이라면, 머신러닝은 데이터에서 규칙을 찾아내는 것입니다.
머신러닝에는 지도 학습, 비지도 학습, 강화 학습이라는 세 가지 주요 범주가 있습니다.
지도 학습은 레이블이 있는 데이터로 모델을 훈련시키는 방법으로, 이미지 분류나 스팸 필터링에 사용됩니다.
비지도 학습은 레이블이 없는 데이터에서 패턴을 찾는 방법으로, 고객 세분화나 이상 탐지에 활용됩니다.
강화 학습은 보상 시스템을 통해 최적의 행동을 학습하는 방법으로, 게임 AI나 로봇 제어에 적용됩니다.

딥러닝은 머신러닝의 한 분야로, 인공 신경망을 사용하여 복잡한 패턴을 학습합니다.
딥러닝의 핵심은 여러 층으로 구성된 신경망으로, 각 층이 점진적으로 더 추상적인 특징을 학습합니다.
합성곱 신경망은 이미지 인식에 특화되어 있으며, 객체 감지와 얼굴 인식에 널리 사용됩니다.
순환 신경망은 시계열 데이터 처리에 적합하며, 자연어 처리와 음성 인식에 활용됩니다.
트랜스포머 아키텍처는 최근 자연어 처리 분야를 혁신했으며, GPT와 BERT 같은 대규모 언어 모델의 기반이 되었습니다.

인공지능 기술은 의료, 금융, 제조, 교통 등 다양한 산업에 혁신을 가져오고 있습니다.
의료 분야에서는 AI가 질병 진단, 신약 개발, 개인 맞춤 치료에 활용되고 있습니다.
금융 분야에서는 사기 탐지, 알고리즘 트레이딩, 신용 평가에 AI가 사용됩니다.
제조업에서는 예측 유지보수, 품질 관리, 공급망 최적화에 AI가 기여하고 있습니다.
자율주행 자동차는 AI 기술의 집합체로, 컴퓨터 비전, 센서 융합, 의사결정 알고리즘이 결합되어 있습니다.

그러나 인공지능의 발전은 윤리적, 사회적 문제도 야기하고 있습니다.
알고리즘 편향은 훈련 데이터의 편향이 AI 시스템에 반영되어 불공정한 결과를 초래할 수 있습니다.
개인정보 보호는 AI가 대량의 개인 데이터를 수집하고 분석하면서 중요한 이슈가 되었습니다.
일자리 대체에 대한 우려도 있지만, 동시에 새로운 직업이 창출되기도 합니다.
AI의 설명 가능성은 특히 의료나 법률 같은 중요한 결정에서 필수적입니다.
인공지능의 안전성과 책임성을 보장하기 위한 규제와 가이드라인이 세계 각국에서 논의되고 있습니다.
"""


def print_summary(summarizer_name, summarizer, parser):
    """요약 결과를 출력하는 함수"""
    print(f"\n{'='*80}")
    print(f"{summarizer_name} 알고리즘")
    print(f"{'='*80}")

    for i, sentence in enumerate(summarizer(parser.document, SENTENCES_COUNT), 1):
        print(f"{i}. {sentence}")


if __name__ == "__main__":
    # 파서 설정
    parser = PlaintextParser.from_string(TEXT, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    print("="*80)
    print("한국어 텍스트 요약 예제 1: 인공지능과 머신러닝 (동일 주제)")
    print("="*80)
    print(f"\n원본 텍스트 문장 수: {len(parser.document.sentences)}")
    print(f"요약할 문장 수: {SENTENCES_COUNT}")

    # 1. Luhn 알고리즘
    summarizer = LuhnSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    print_summary("Luhn", summarizer, parser)

    # 2. LSA (Latent Semantic Analysis) 알고리즘
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    print_summary("LSA (Latent Semantic Analysis)", summarizer, parser)

    # 3. LexRank 알고리즘
    summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    print_summary("LexRank", summarizer, parser)

    # 4. TextRank 알고리즘
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    print_summary("TextRank", summarizer, parser)

    # 5. SumBasic 알고리즘
    summarizer = SumBasicSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    print_summary("SumBasic", summarizer, parser)

    # 6. KL-Sum 알고리즘
    summarizer = KLSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    print_summary("KL-Sum", summarizer, parser)

    # 7. Reduction 알고리즘
    summarizer = ReductionSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    print_summary("Reduction", summarizer, parser)

    print("\n" + "="*80)
    print("요약 완료!")
    print("="*80)
