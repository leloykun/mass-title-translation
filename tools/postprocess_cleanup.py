import sys, regex, opencc, unicodedata, six
import pandas as pd

def cleanup(line):
    line = line.replace('  & # 39 ; ', '\'')
    line = line.replace('& amp ;', '&')
    line = line.replace('& quot ;', '\"')
    line = line.replace('& lt ;', '<')
    line = line.replace('& gt ;', '>')
    return line

for line in sys.stdin:
    line = cleanup(line)
    print(u'%s' % line, end="")
