#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대용량 텍스트 데이터 다운로드 및 생성
Wikipedia API를 사용하여 긴 문서들을 다운로드
"""

import requests
import time
import sys

def download_wikipedia_article(title, lang='en'):
    """Wikipedia 문서 다운로드"""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        pages = data['query']['pages']

        for page_id in pages:
            if 'extract' in pages[page_id]:
                return pages[page_id]['extract']
        return None
    except Exception as e:
        print(f"Error downloading {title}: {e}")
        return None


def main():
    # 긴 Wikipedia 문서들 (영어)
    articles = [
        "History_of_the_United_States",
        "World_War_II",
        "Earth",
        "Evolution",
        "Climate_change",
        "Artificial_intelligence",
        "Computer_science",
        "Physics",
        "Chemistry",
        "Biology",
        "Mathematics",
        "Philosophy",
        "Psychology",
        "Economics",
        "History_of_science",
        "Ancient_Rome",
        "Ancient_Greece",
        "Renaissance",
        "Industrial_Revolution",
        "Space_exploration",
        "Quantum_mechanics",
        "Theory_of_relativity",
        "DNA",
        "Human_brain",
        "Solar_System",
        "Milky_Way",
        "Big_Bang",
        "Black_hole",
        "Periodic_table",
        "Organic_chemistry",
        "Cellular_biology",
        "Genetics",
        "Biodiversity",
        "Ecology",
        "Geography",
        "World_history",
        "Medieval_history",
        "Modern_history",
        "Technology",
        "Internet",
        "Machine_learning",
        "Deep_learning",
        "Natural_language_processing",
        "Robotics",
        "Astronomy",
        "Astrophysics",
        "Cosmology",
        "Geology",
        "Oceanography",
        "Meteorology",
    ]

    output_file = "large_wikipedia_text.txt"
    total_chars = 0
    total_mb = 0

    print(f"Downloading {len(articles)} Wikipedia articles...")
    print(f"Target: ~100MB+ text data")
    print("=" * 80)

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, article in enumerate(articles, 1):
            print(f"\n[{i}/{len(articles)}] Downloading: {article}")

            text = download_wikipedia_article(article.replace('_', ' '))

            if text:
                f.write(f"\n\n{'='*80}\n")
                f.write(f"ARTICLE: {article.replace('_', ' ')}\n")
                f.write(f"{'='*80}\n\n")
                f.write(text)

                chars = len(text)
                total_chars += chars
                total_mb = total_chars / (1024 * 1024)

                print(f"  Size: {chars:,} chars ({chars/1024/1024:.2f} MB)")
                print(f"  Total so far: {total_chars:,} chars ({total_mb:.2f} MB)")

                # 100MB 이상이면 중단
                if total_mb >= 100:
                    print(f"\n✓ Reached target size: {total_mb:.2f} MB")
                    break
            else:
                print(f"  ✗ Failed to download")

            # API 호출 제한을 피하기 위해 잠시 대기
            time.sleep(1)

    print("\n" + "="*80)
    print(f"✓ Download complete!")
    print(f"  File: {output_file}")
    print(f"  Total size: {total_chars:,} characters ({total_mb:.2f} MB)")
    print(f"  Articles downloaded: {i}")
    print("="*80)


if __name__ == "__main__":
    main()
