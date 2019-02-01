#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wj
"""
import re
#normal digits
cn_digit_01_str='零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟0123456789'
cn_digit_01_set=set(list(cn_digit_01_str))
cn_digit_01_dict={'零':'0','一':'1','二':'2','三':'3','四':'4','五':'5',
                  '六':'6','七':'7','八':'8','九':'9',
                  '壹':'1','贰':'2','叁':'3','肆':'4','伍':'5','陆':'6','柒':'7','捌':'8',
                  '玖':'9','拾':'十','佰':'百','仟':'千'}
#phone call digits
cn_digit_dict={'幺':'1','零':'0','一':'1','二':'2','三':'3','四':'4','五':'5',
               '六':'6','七':'7','八':'8','九':'9'}
digit_cn_dict={'0':'零','1':'一','2':'二','3':'三','4':'四','5':'五',
               '6':'六','7':'七','8':'八','9':'九'}

def reg_thousand(raw_number,end=False):
    
    if not isinstance(raw_number,str):
        return ''
    if len(raw_number)<1:
        if end:
            return '0000'
        return ''
    for key in cn_digit_01_dict:
        raw_number=raw_number.replace(key,cn_digit_01_dict[key])
    di_num=raw_number
    if len(re.findall("十",raw_number))>1 or len(re.findall('百',raw_number))>1 or len(re.findall('千',raw_number))>1:
        return ''
    thousand='0'
    hundred='0'
    ten='0'
    digit='0'
    if re.search("千",di_num):
        th_content=di_num.split("千")[0]
        di_num=di_num.split("千")[1]
        th_num=re.search("^0?([1-9])$",th_content)
        if th_num:
            thousand=th_num.group(1)
        else:
            return ''
    if re.search("百",di_num):
        hu_content=di_num.split("百")[0]
        di_num=di_num.split("百")[1]
        hu_num=re.search("^0?([1-9])$",hu_content)
        if hu_num:
            hundred=hu_num.group(1)
        else:
            return ''
    if re.search("十",di_num):
        ten_content=di_num.split('十')[0]
        di_num=di_num.split('十')[1]
        if len(ten_content)<1:
            ten='1'
        else:
            ten_num=re.search("^0?([1-9])$",ten_content)
            if ten_num:
                ten=ten_num.group(1)
            else:
                return ''
    if re.search("^0([1-9])$",di_num):
        digit=re.search("^0([1-9])$",di_num).group(1)
    elif di_num=='0' and (not re.search("十|百|千",raw_number)):
        digit='0'
    elif re.search("^[1-9]$",di_num):
        if re.search("十",raw_number):
            digit=re.search("^[1-9]$",di_num).group()
        elif re.search("百",raw_number):
            ten=re.search("^[1-9]$",di_num).group()
        elif re.search("千",raw_number):
            hundred=re.search("^[1-9]$",di_num).group()
        else:
            if end:
                thousand=re.search("^[1-9]$",di_num).group()
            else:
                digit=re.search("^[1-9]$",di_num).group()
    elif di_num=='':
        if not re.search("十|百|千",raw_number):
            return ''
    else:
        return ''
    thousand=int(thousand)
    hundred=int(hundred)
    ten=int(ten)
    digit=int(digit)
    reg_digits=1000*thousand+100*hundred+10*ten+digit
    reg_digits=str(reg_digits)
    #reg_digits="0"*(4-len(reg_digits))+reg_digits
    return reg_digits

#convert '.' between two digit  to '点'
def convert_dot_cn(sentence):
    new_sentence=[]
    for i,s in enumerate(sentence):
        #print(i,s)
        if s=='.':
            if i>0 and i<len(sentence)-1 and sentence[i-1] in cn_digit_01_set and sentence[i+1] in cn_digit_01_set:
                new_sentence.append('点')
            else:
                new_sentence.append(s)
        else:
            new_sentence.append(s)
    return "".join(new_sentence)
#convert number to <number> which greater than ten thousand.
def  convert_pure_digits(sentence):
    #convert cn digit to Arabic
    sentence_digit="".join([cn_digit_dict.get(i,i) for i in sentence])
    sentence_split=re.split("([0-9]+)",sentence_digit)
    #print(sentence_split)
    new_sentence=''
    for s in sentence_split:
        if s:
            if s.isdigit():
                if len(s)==1:
                    new_sentence+=digit_cn_dict[s]
                else:
                    if len(s)>4:
                        new_sentence+=' <number> '
                    else:
                        new_sentence+=" "+s+" "
            else:
                new_sentence+=s
    return new_sentence
def convert_complex_digits(sentence):
    number_set='零一二三四五六七八九十百千壹贰叁肆伍陆柒捌玖拾佰仟0123456789'  
    sentence_split=re.split("([{}]+)".format(number_set),sentence)
    new_sentence=[]
    for s in sentence_split:
        if len(s)>1 and re.search('^[{}]+$'.format(number_set),s):
            c_s=reg_thousand(s)
            if c_s:
                new_sentence.append(str(c_s))
            else:
                new_sentence.append(s)
        else:
            new_sentence.append(s)
    return " ".join(new_sentence)
#character level tokenization
#step 1: process '点.', '.' between two digit convert to '点'
#step 2: convert number to Arabic numerals (less than ten thousand, length more than 1)
#       convert number to cn numerals(length=1)
#       convert number to <number> which greater than ten thousand
import unicodedata
def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False
def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
class Tokenizer:
    def __init__(self):
        pass
    def tokenize(self,text,do_lower_case=True):
        #text=self.cn_digit_tokenize(text)
        #delete control character (\t \n \r not comtained)
        #convert a white space as ' '
        text=self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
    def cn_digit_tokenize(self,sentence,digit_min_length=4,contain_space=True):
        if contain_space:
            digits_list='[幺零一二三四五六七八九0123456789 ]+'
        else:
            digits_list='[幺零一二三四五六七八九0123456789]+'
        sentence_split=re.split("({})".format(digits_list),sentence)
        new_sentence=''
        for s in sentence_split:
            if re.search("^{}$".format(digits_list),s):
                if len(s.replace(" ",''))>digit_min_length:
                    new_sentence+=" cndigits "
                else:
                    new_sentence+=s
            else:
                new_sentence+=s
        #for key in digit_cn_dict:
        #    new_sentence=new_sentence.replace(key,digit_cn_dict[key])
        return new_sentence
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True
        return False
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]
    def load_vocab_tags(self,vocab_file,tag_file):
        self.vocab={}
        self.tags={}
        with open(vocab_file,'r') as f:
            for line in f:
                line=line.strip()
                if not line :
                    continue
                if line not in self.vocab:
                    self.vocab[line]=len(self.vocab)
        with open(tag_file,'r') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                if line not in self.tags:
                    self.tags[line]=len(self.tags)
    def convert_tokens_to_ids(self,tokens):
        ids=[]
        for t in tokens:
            ids.append(self.vocab.get(t,'<unk>'))
        return ids
    def convert_tag_to_ids(self,tag):
        return self.tags[tag]

