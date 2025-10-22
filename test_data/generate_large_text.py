#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대용량 텍스트 파일 생성 (1GB)
실제 의미있는 한국어 텍스트를 반복
"""

import time

# 한국어 예제 1의 텍스트
TEXT1 = """
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

# 한국어 예제 2의 텍스트
TEXT2 = """
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
"""

def generate_large_file(target_size_gb=1.0, output_file="large_text.txt"):
    """대용량 텍스트 파일 생성"""

    target_size_bytes = int(target_size_gb * 1024 * 1024 * 1024)

    # 두 텍스트를 결합
    combined_text = TEXT1 + "\n\n" + TEXT2 + "\n\n"
    text_size = len(combined_text.encode('utf-8'))

    # 필요한 반복 횟수 계산
    iterations_needed = target_size_bytes // text_size + 1

    print(f"Generating {target_size_gb} GB text file...")
    print(f"Text block size: {text_size:,} bytes ({text_size/1024/1024:.2f} MB)")
    print(f"Iterations needed: {iterations_needed:,}")
    print(f"Target size: {target_size_bytes:,} bytes ({target_size_gb:.2f} GB)")
    print("=" * 80)

    start_time = time.time()
    current_size = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(iterations_needed):
            f.write(combined_text)
            current_size += text_size

            # 진행상황 출력 (10% 단위)
            progress = (current_size / target_size_bytes) * 100
            if i % max(1, iterations_needed // 10) == 0:
                elapsed = time.time() - start_time
                mb_written = current_size / 1024 / 1024
                speed = mb_written / elapsed if elapsed > 0 else 0
                print(f"Progress: {progress:.1f}% | "
                      f"Written: {mb_written:.1f} MB | "
                      f"Speed: {speed:.1f} MB/s | "
                      f"Elapsed: {elapsed:.1f}s")

            # 목표 크기 도달
            if current_size >= target_size_bytes:
                break

    end_time = time.time()
    elapsed = end_time - start_time
    final_size_mb = current_size / 1024 / 1024
    final_size_gb = current_size / 1024 / 1024 / 1024

    print("\n" + "=" * 80)
    print("✓ File generation complete!")
    print(f"  File: {output_file}")
    print(f"  Size: {current_size:,} bytes ({final_size_mb:.2f} MB / {final_size_gb:.3f} GB)")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Speed: {final_size_mb/elapsed:.2f} MB/s")
    print("=" * 80)


if __name__ == "__main__":
    # 1GB 텍스트 파일 생성
    generate_large_file(target_size_gb=1.0, output_file="large_text_1gb.txt")
