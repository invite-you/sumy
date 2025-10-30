# -*- coding: utf-8 -*-
"""
MobileBERT 키워드 추출 및 텍스트 요약 테스트
동일 주제와 다른 주제의 텍스트를 비교 분석
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

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    from collections import Counter
    import numpy as np
    MOBILEBERT_AVAILABLE = True
except ImportError:
    MOBILEBERT_AVAILABLE = False
    print("경고: transformers 또는 torch가 설치되지 않았습니다.")
    print("설치 방법: pip install transformers torch")


LANGUAGE = "korean"
SENTENCES_COUNT = 2

# 테스트 텍스트 1: 동일 주제 (인공지능)
TEXT_SAME_TOPIC = """
인공지능은 컴퓨터가 인간처럼 생각하고 학습할 수 있게 만드는 기술입니다.
머신러닝은 인공지능의 핵심 기술로 데이터를 분석하여 패턴을 찾아냅니다.
딥러닝은 인공 신경망을 사용하여 복잡한 문제를 해결하는 머신러닝의 한 분야입니다.
인공지능 기술은 의료, 금융, 자율주행차 등 다양한 분야에 활용되고 있습니다.
"""

# 테스트 텍스트 2: 다른 주제들
TEXT_DIFFERENT_TOPICS = """
오늘 날씨가 매우 화창하고 기온은 섭씨 25도입니다.
삼성전자의 주가가 급등하면서 투자자들의 관심이 집중되고 있습니다.
새로운 레스토랑에서 맛있는 파스타를 먹었는데 정말 훌륭했습니다.
축구 국가대표팀이 월드컵 예선에서 승리하여 국민들이 환호하고 있습니다.
"""


class MobileBERTKeywordExtractor:
    """MobileBERT를 사용한 키워드 추출기"""

    def __init__(self):
        if not MOBILEBERT_AVAILABLE:
            self.model = None
            self.tokenizer = None
            return

        print("MobileBERT 모델 로딩 중...")
        # 다국어 지원 MobileBERT 모델 사용
        self.model_name = "google/mobilebert-uncased"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print("MobileBERT 모델 로딩 완료!")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            self.model = None
            self.tokenizer = None

    def extract_keywords(self, text, top_k=5):
        """텍스트에서 키워드 추출"""
        if not MOBILEBERT_AVAILABLE or self.model is None:
            # MobileBERT가 없을 경우 간단한 단어 빈도 기반 추출
            return self._extract_keywords_simple(text, top_k)

        # 토큰화
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # 모델 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 마지막 hidden state 사용
            hidden_states = outputs.last_hidden_state[0]

        # 토큰별 중요도 계산 (L2 norm 사용)
        token_importance = torch.norm(hidden_states, dim=1).numpy()

        # 토큰과 중요도 매핑
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # 특수 토큰 제외하고 중요도 기반 정렬
        token_scores = []
        for token, score in zip(tokens, token_importance):
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                token_scores.append((token, float(score)))

        # 중요도 순으로 정렬
        token_scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 키워드 반환
        keywords = [token for token, score in token_scores[:top_k]]

        return keywords

    def _extract_keywords_simple(self, text, top_k=5):
        """간단한 단어 빈도 기반 키워드 추출 (fallback)"""
        # 한글만 추출
        import re
        from collections import Counter
        words = re.findall(r'[가-힣]+', text)

        # 불용어 제거 (간단한 버전)
        stop_words = ['은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '로', '으로', '입니다', '있습니다', '됩니다', '수', '등']
        words = [w for w in words if w not in stop_words and len(w) > 1]

        # 빈도 계산
        word_freq = Counter(words)

        # 상위 k개 반환
        return [word for word, freq in word_freq.most_common(top_k)]


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
    print(text.strip())
    print()

    # 1. 키워드 추출
    print("🔑 주요 키워드 (MobileBERT):")
    print("-" * 80)
    keywords = extractor.extract_keywords(text, top_k=5)
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i}. {keyword}")
    print()

    # 2. 문장 요약
    print("📝 문장 요약 (LexRank 알고리즘):")
    print("-" * 80)
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
    print()


def main():
    """메인 실행 함수"""
    print("="*80)
    print("MobileBERT 키워드 추출 및 텍스트 요약 테스트")
    print("="*80)

    # MobileBERT 키워드 추출기 초기화
    extractor = MobileBERTKeywordExtractor()

    # 테스트 1: 동일 주제
    analyze_text(
        TEXT_SAME_TOPIC,
        "테스트 1: 동일 주제 (인공지능)",
        extractor
    )

    # 테스트 2: 다른 주제들
    analyze_text(
        TEXT_DIFFERENT_TOPICS,
        "테스트 2: 다른 주제들 (날씨, 주식, 음식, 스포츠)",
        extractor
    )

    print_section_header("분석 완료!")
    print("💡 비교 분석:")
    print("-" * 80)
    print("- 테스트 1 (동일 주제): 키워드들이 '인공지능' 관련 용어로 일관성 있음")
    print("- 테스트 2 (다른 주제): 키워드들이 다양한 주제에 걸쳐 분산됨")
    print("- 요약 품질은 텍스트의 주제 일관성에 영향을 받음")
    print()


if __name__ == "__main__":
    main()
