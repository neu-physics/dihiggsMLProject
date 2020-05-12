## command line: python3.7 EFN.py --sigInput pp2hh4b_1M_dict.txt --bkgInput ppTo4b_4M_dict.txt --nSig 600000 --nBkg 600000 --nJets 1 --nTags 0
import pickle
import numpy as np
import argparse
import subprocess

import sys
#sys.path.insert(0, '/afs/cern.ch/work/l/lian/public/diHiggs/dihiggsMLProject/')
sys.path.insert(0, '/uscms/home/ali/nobackup/diHiggs/dihiggsMLProject/')
from utils.commonFunctions import *

parser = argparse.ArgumentParser()
parser.add_argument('--sigInput',dest='sigInput')
parser.add_argument('--bkgInput',dest='bkgInput')
parser.add_argument('--nSig',dest='nSig',type=int, default=-1)
parser.add_argument('--nBkg',dest='nBkg',type=int, default=-1)
parser.add_argument('--nJets',dest='nJets', type=int, default=1)
parser.add_argument('--nTags',dest='nTags', type=int, default=0)

args = parser.parse_args()

nsig = 0
nbkg = 0
sig_list = []
bkg_list = []
with open(args.sigInput) as sig_path:
  line = sig_path.readline()
  while (line and ( (args.nSig>=0 and nsig<args.nSig) or (args.nSig==-1) ) ):
    f_path = line.strip('\n')
    f_sig = open(f_path,'rb')
    sig = pickle.load(f_sig)
    f_sig.close()

    sig_list.extend(sig)
    line = sig_path.readline()
    nsig += len(sig)

with open(args.bkgInput) as bkg_path:
  line = bkg_path.readline()
  while (line and ( (args.nBkg>=0 and nbkg<args.nBkg) or (args.nBkg==-1) ) ):
    f_path = line.strip('\n')
    f_bkg = open(f_path,'rb')
    bkg = pickle.load(f_bkg)
    f_bkg.close()

    bkg_list.extend(bkg)
    line = bkg_path.readline()
    nbkg += len(bkg)

print("Finished loading data :)")
print("# sig: {0}  # bkg: {1}".format(nsig,nbkg))

n_sigPassCut = 0
n_bkgPassCut = 0

for evt in sig_list:
  if(evt['nJets']>=args.nJets and evt['nBTags']>=args.nTags ):
    n_sigPassCut += 1

for evt in bkg_list:
  if(evt['nJets']>=args.nJets and evt['nBTags']>=args.nTags):
    n_bkgPassCut += 1


print("# sig passing cut: {}".format(n_sigPassCut))
print("# bkg passing cut: {}".format(n_bkgPassCut))

