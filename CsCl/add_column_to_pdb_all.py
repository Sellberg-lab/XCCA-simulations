#*************************************************************************
# Adding missing column # 78 with 1st letter in Atom for
# reading pdb-files with condor v1.0 (isong spsim)
# 2019-01-21 v3 all 2M,4M,6M-files @ Caroline Dahlqvist cldah@kth.se
# Retrieve from CsCl-PDB-folder
# Save in CsCl_PDB_ed -folder
#*************************************************************************
import os #os = for saving in specific dir;
this_dir = os.path.dirname(os.path.realpath(__file__)) # from Condor/example.py, get run-directory

import string
conc = [2,4,6]
for c in conc:
    for i in range(91):
        outputfile = open(this_dir+'/CsCl-PDB_ed/%iM%i_ed.pdb' %(c,i), "w")
        for line in open(this_dir+'/CsCl-PDB/%iM%i.pdb' %(c,i), "r"):
            #print "length of row", len(line)
            if len(line)==79: print >> outputfile, line[0:77] + line [13]
            else: print >> outputfile, line
        outputfile.close()

#       E.g. print to file:
# >>> with open('spamspam.txt', 'w', opener=opener) as f:
#...     print('This will be written to somedir/spamspam.txt', file=f)
# for python3 (in Python2 must import: from __future__ import print_function)
#       or use 'write()':
#f.write("This is line %d\r\n" % (i+1))
