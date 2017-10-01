import sys, os
old_stdout = sys.stdout
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = old_stdout