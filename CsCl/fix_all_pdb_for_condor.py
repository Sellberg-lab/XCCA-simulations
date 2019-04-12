#*************************************************************************
# Adding missing column # 78 with 1st letter in Atom for
# reading pdb-files with condor v1.0 (isong spsim)
# 2019-01-21 v3 all 2M,4M,6M-files @ Caroline Dahlqvist cldah@kth.se
# Retrieve from CsCl-PDB-folder
# Save in CsCl_PDB_ed -folder
#*************************************************************************
import string
import os #os = for saving in specific dir;
this_dir = os.path.dirname(os.path.realpath(__file__)) # from Condor/example.py, get run-directory

outdir = this_dir+'/CsCl-PDB_Clmn'
if not os.path.exists(outdir):
		os.makedirs(outdir)


conc = [2,4,6]
for c in conc:
    for i in range(91):
        outputfile = open(outdir+'/%iM%i_ed.pdb' %(c,i), "w")
        for line in open(this_dir+'/CsCl-PDB/%iM%i.pdb' %(c,i), "r"):
            if len(line)==79:
            	if line [13] == 'C': 
            		print >> outputfile, line[0:76] + line [13] + line[14].lower()
                    #print >> outputfile, line[0:76] + line [13] + line[14].upper()
            	else: print >> outputfile, line[0:77] + line [13]
                #if line[14]=='W': print >> outputfile, line[0:77] + line [13]
                #else: print >> outputfile, line[0:76] + line [13] + line[14].lower()
            else: print >> outputfile, line
        outputfile.close()

#       E.g. print to file:
# >>> with open('spamspam.txt', 'w', opener=opener) as f:
#...     print('This will be written to somedir/spamspam.txt', file=f)
# for python3 (in Python2 must import: from __future__ import print_function)
#       or use 'write()':
#f.write("This is line %d\r\n" % (i+1))
