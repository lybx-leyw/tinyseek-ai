"""
本工具定义了一个分词器
"""

import re
# import jieba

class Tokenizer():
    def __init__(self):
        pass

    def tokenize(text,JIEBA=False):
        if JIEBA is True:
            # tokens = jieba.lcut(text) 该功能已弃用
            JIEBA = False

        tokens = re.findall(r'<\|\s*im_end\s*\|>|[a-zA-Z]+|\d+(?:\.\d+)?%?|[^\w\s]|[\u4e00-\u9fff]',text)
        
        result = []
        for token in tokens:
            if re.search(r'\d', token) or '.' in token or '%' in token:
                result.extend(list(token))
            else:
                result.append(token)
        
        return result