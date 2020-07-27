import sys, regex, opencc, unicodedata, six
import pandas as pd

p = regex.compile(r'\p{So}')
converter_tw2s = opencc.OpenCC('tw2s.json')
converter_t2s = opencc.OpenCC('t2s.json')
converter_hk2s = opencc.OpenCC('hk2s.json')

def cleanup(line):
    line = line.replace('\n', ' ')
    line = line.replace('\"', ' ')
    line = line.replace(',', ' ')
    line = line.replace(' & # 39 ; ', '\'')
    line = line.replace('& amp ;', '&')
    line = line.replace('& quot ;', '\"')
    line = p.sub(" ", line)
    return line

def simplify(line):
    line = converter_tw2s.convert(line)
    line = converter_t2s.convert(line)
    line = converter_hk2s.convert(line)
    return line

def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding='utf-8', errors='strict'):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")

def run_strip_accents(text):
    """
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


for line in sys.stdin:
    line = cleanup(line)
    line = simplify(line)
    line = convert_to_unicode(line.rstrip().lower())
    line = run_strip_accents(line)
    line = line.lower()
    print(u'%s' % line)
