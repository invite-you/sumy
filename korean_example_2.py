# -*- coding: utf-8 -*-
"""
한국어 예제 2: 다양한 주제의 긴 내용 요약
주제: 과학, 환경, 경제, 문화 등 다양한 분야
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
SENTENCES_COUNT = 4

# 다양한 주제가 섞인 긴 텍스트
TEXT = """
양자 컴퓨터는 양자역학의 원리를 이용하여 정보를 처리하는 차세대 컴퓨팅 기술입니다.
기존의 컴퓨터가 0과 1의 비트를 사용하는 반면, 양자 컴퓨터는 중첩 상태에 있을 수 있는 큐비트를 사용합니다.
양자 얽힘과 중첩 현상을 활용하면 특정 문제에 대해 기존 컴퓨터보다 기하급수적으로 빠른 연산이 가능합니다.
구글과 IBM을 비롯한 주요 기업들이 양자 컴퓨터 개발에 막대한 투자를 하고 있습니다.
암호학, 신약 개발, 금융 모델링 등에서 양자 컴퓨터의 활용이 기대되고 있습니다.

기후 변화는 21세기 인류가 직면한 가장 큰 도전 중 하나입니다.
산업혁명 이후 화석 연료 사용이 증가하면서 대기 중 이산화탄소 농도가 급격히 상승했습니다.
지구 평균 온도는 지난 100년간 약 1도 상승했으며, 이로 인해 극지방의 빙하가 녹고 해수면이 상승하고 있습니다.
기후 변화는 생태계를 파괴하고, 극단적인 기상 현상을 빈번하게 만들며, 식량 안보를 위협합니다.
국제사회는 파리 기후협정을 통해 지구 온난화를 산업화 이전 대비 1.5도 이내로 제한하기로 합의했습니다.
재생 가능 에너지로의 전환, 에너지 효율 향상, 탄소 포집 기술 개발이 시급한 과제입니다.

블록체인 기술은 분산 원장 기술로서 중앙 관리자 없이 데이터를 안전하게 저장하고 관리할 수 있게 합니다.
비트코인으로 시작된 블록체인은 이제 금융, 물류, 의료, 부동산 등 다양한 분야로 확산되고 있습니다.
스마트 계약은 블록체인 위에서 자동으로 실행되는 계약으로, 중개자 없이 신뢰할 수 있는 거래를 가능하게 합니다.
탈중앙화 금융은 전통적인 금융 중개자 없이 대출, 예금, 거래 등의 금융 서비스를 제공합니다.
그러나 블록체인의 확장성 문제, 에너지 소비, 규제 이슈 등은 아직 해결해야 할 과제입니다.

한국의 전통 문화유산은 오랜 역사를 통해 형성된 독특한 가치를 지니고 있습니다.
한글은 세종대왕이 창제한 과학적이고 체계적인 문자로, 유네스코 세계기록유산에 등재되어 있습니다.
한국의 전통 음악인 국악은 정악과 민속악으로 나뉘며, 각기 다른 아름다움을 표현합니다.
판소리는 한 명의 소리꾼이 고수의 장단에 맞춰 긴 이야기를 노래로 풀어내는 예술 형식입니다.
한복은 선의 아름다움과 색의 조화를 중시하는 전통 의상으로, 현대적으로 재해석되어 일상복으로도 사랑받고 있습니다.

우주 탐사는 인류의 오랜 꿈이자 과학 기술의 최전선입니다.
NASA의 제임스 웹 우주 망원경은 우주의 초기 모습을 관찰하여 빅뱅 이후의 은하 형성을 연구하고 있습니다.
화성 탐사 로버들은 화성 표면을 탐사하며 과거 물의 존재와 생명체 흔적을 찾고 있습니다.
SpaceX와 같은 민간 우주 기업들은 재사용 가능한 로켓 기술로 우주 여행 비용을 획기적으로 낮추고 있습니다.
인류는 달 기지 건설과 화성 유인 탐사를 목표로 하며, 우주 자원 채굴도 검토하고 있습니다.

코로나19 팬데믹은 전 세계적으로 공중 보건 시스템과 경제에 큰 영향을 미쳤습니다.
바이러스의 급속한 확산은 국경을 봉쇄하고 사회적 거리두기를 필요로 했습니다.
mRNA 백신 기술의 성공은 의학 발전의 중요한 이정표가 되었으며, 향후 다른 질병 치료에도 응용될 것입니다.
팬데믹은 원격 근무와 온라인 교육을 가속화하여 일하고 배우는 방식을 변화시켰습니다.
공급망 붕괴와 인플레이션은 경제적 불안정을 야기했고, 각국 정부는 대규모 경기 부양책을 실시했습니다.

유전자 편집 기술인 CRISPR은 생명공학 분야에 혁명을 일으키고 있습니다.
CRISPR-Cas9 시스템은 DNA의 특정 부위를 정확하게 잘라내고 편집할 수 있게 해줍니다.
유전 질환 치료, 농작물 개량, 질병 저항성 동물 개발 등 다양한 응용이 가능합니다.
그러나 인간 배아 유전자 편집에 대한 윤리적 논란이 계속되고 있습니다.
과학계는 유전자 편집의 안전성과 장기적 영향에 대한 신중한 연구가 필요하다고 강조합니다.

세계 경제는 디지털 전환과 지정학적 불확실성 속에서 재편되고 있습니다.
인공지능과 자동화는 생산성을 높이지만 노동 시장의 양극화를 심화시킬 수 있습니다.
전자상거래의 급성장은 전통적인 소매업을 변화시키고 물류 산업을 확대시켰습니다.
중국과 미국 간의 기술 패권 경쟁은 글로벌 공급망의 재편을 촉진하고 있습니다.
ESG 투자는 환경, 사회, 지배구조를 고려한 지속 가능한 경영을 요구하며 기업 평가의 새로운 기준이 되고 있습니다.
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
    print("한국어 텍스트 요약 예제 2: 다양한 주제 (과학, 환경, 경제, 문화 등)")
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
