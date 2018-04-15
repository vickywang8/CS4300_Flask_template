import numpy as np
import math
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
tokenize_blurb = tokenizer.tokenize(blurb)

