#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re

fname = sys.argv[1]
with open(fname) as f:
    for line in f.readlines():
        content_cleanup = re.sub(r'\+\|[A-Za-z0-9．.!]+\||\n|\[.+\]?', '', line)
        if not (content_cleanup[-5:-1] == ".wav"):
            print content_cleanup
'''
content_search = re.search("\"content\": \".+\",", data, re.IGNORECASE)
print dir(content_search)
print content_search.group(0)
'''
