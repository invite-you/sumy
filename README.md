# Automatic text summarizer


[![image](https://github.com/miso-belica/sumy/actions/workflows/run-tests.yml/badge.svg)](https://github.com/miso-belica/sumy/actions/workflows/run-tests.yml)
[![GitPod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/miso-belica/sumy) 

Simple library and command line utility for extracting summary from HTML
pages or plain texts. The package also contains simple evaluation
framework for text summaries. Implemented summarization methods are described in the [documentation](docs/summarizators.md). I also maintain a list of [alternative implementations](docs/alternatives.md) of the summarizers in various programming languages.

## Is my natural language supported?
There is a [good chance](docs/index.md#Tokenizer) it is. But if not it is [not too hard to add](docs/how-to-add-new-language.md) it.

## Installation

Make sure you have [Python](http://www.python.org/) 3.6+ and
[pip](https://crate.io/packages/pip/)
([Windows](http://docs.python-guide.org/en/latest/starting/install/win/),
[Linux](http://docs.python-guide.org/en/latest/starting/install/linux/))
installed. Run simply (preferred way):

```sh
$ [sudo] pip install sumy
$ [sudo] pip install git+git://github.com/miso-belica/sumy.git  # for the fresh version
```

## Usage

Thanks to some good soul out there, the easiest way to try sumy is in your browser at https://huggingface.co/spaces/issam9/sumy_space

Sumy contains command line utility for quick summarization of documents.

```sh
$ sumy lex-rank --length=10 --url=https://en.wikipedia.org/wiki/Automatic_summarization # what's summarization?
$ sumy lex-rank --language=uk --length=30 --url=https://uk.wikipedia.org/wiki/Україна
$ sumy luhn --language=czech --url=https://www.zdrojak.cz/clanky/automaticke-zabezpeceni/
$ sumy edmundson --language=czech --length=3% --url=https://cs.wikipedia.org/wiki/Bitva_u_Lipan
$ sumy --help # for more info
```

Various evaluation methods for some summarization method can be executed
by commands below:

```sh
$ sumy_eval lex-rank reference_summary.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization
$ sumy_eval lsa reference_summary.txt --language=czech --url=https://www.zdrojak.cz/clanky/automaticke-zabezpeceni/
$ sumy_eval edmundson reference_summary.txt --language=czech --url=https://cs.wikipedia.org/wiki/Bitva_u_Lipan
$ sumy_eval --help # for more info
```

If you don't want to bother by the installation, you can try it as a container.

```sh
$ docker run --rm misobelica/sumy lex-rank --length=10 --url=https://en.wikipedia.org/wiki/Automatic_summarization
```

## Python API

Or you can use sumy like a library in your project. Create file `sumy_example.py` ([don't name it `sumy.py`](https://stackoverflow.com/questions/41334622/python-sumy-no-module-named-sumy-parsers-html)) with the code below to test it.

```python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


LANGUAGE = "english"
SENTENCES_COUNT = 10


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Automatic_summarization"
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    # parser = PlaintextParser.from_string("Check this out.", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)
```

## Korean Language Example

Sumy supports Korean text summarization with high-performance tokenizers. Here are complete examples using all available summarization algorithms.

### Example 1: Single Topic (AI and Machine Learning)

See [`korean_example_1.py`](korean_example_1.py) for a complete example that summarizes a long text about artificial intelligence and machine learning using all 7 summarization algorithms (Luhn, LSA, LexRank, TextRank, SumBasic, KL-Sum, and Reduction).

**Requirements:**
```sh
$ pip install sumy kiwipiepy
$ python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

**Quick start:**
```python
# -*- coding: utf-8 -*-
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "korean"
SENTENCES_COUNT = 3

TEXT = """
인공지능은 컴퓨터 과학의 한 분야로서 기계가 인간과 유사한 지능을 가지도록 만드는 기술입니다.
인공지능의 역사는 1950년대로 거슬러 올라가며, 앨런 튜링이 기계가 사고할 수 있는지에 대한 질문을 제기하면서 시작되었습니다.
머신러닝은 인공지능의 핵심 기술 중 하나로, 데이터로부터 패턴을 학습하여 예측이나 결정을 내리는 방법입니다.
딥러닝은 머신러닝의 한 분야로, 인공 신경망을 사용하여 복잡한 패턴을 학습합니다.
인공지능 기술은 의료, 금융, 제조, 교통 등 다양한 산업에 혁신을 가져오고 있습니다.
"""

parser = PlaintextParser.from_string(TEXT, Tokenizer(LANGUAGE))
stemmer = Stemmer(LANGUAGE)

summarizer = LexRankSummarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)
```

**Run the complete example:**
```sh
$ python korean_example_1.py
```

### Example 2: Multiple Topics (Science, Environment, Economy, Culture)

See [`korean_example_2.py`](korean_example_2.py) for an example that summarizes a long text covering diverse topics including quantum computing, climate change, blockchain, Korean culture, space exploration, COVID-19, gene editing, and global economy.

**Run the complete example:**
```sh
$ python korean_example_2.py
```

Both examples demonstrate all 7 summarization algorithms:
1. **Luhn** - Based on word frequency and sentence significance
2. **LSA (Latent Semantic Analysis)** - Uses matrix decomposition to find topics
3. **LexRank** - Graph-based algorithm using sentence similarity
4. **TextRank** - Similar to PageRank, measures sentence importance
5. **SumBasic** - Selects sentences based on word probability
6. **KL-Sum** - Uses KL divergence to measure sentence importance
7. **Reduction** - Removes less important sentences iteratively

## Interesting projects using sumy

I found some interesting projects while browsing the internet or sometimes people wrote me an e-mail with questions, and I was curious how they use the sumy :)

* [Learning to generate questions from text](https://software.intel.com/en-us/articles/using-natural-language-processing-for-smart-question-generation) - https://github.com/adityasarvaiya/Automatic_Question_Generation
* Summarize your video to any duration - https://github.com/aswanthkoleri/VideoMash and similar https://github.com/OpenGenus/vidsum
* Tool for collectively summarizing large discussions - https://github.com/amyxzhang/wikum
* AutoTL;DR bot for [Lemmy](https://en.wikipedia.org/wiki/Lemmy_(software)) uses sumy: https://github.com/RikudouSage/LemmyAutoTldrBot
