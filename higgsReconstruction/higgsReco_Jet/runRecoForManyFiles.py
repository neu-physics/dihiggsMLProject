import os,sys,argparse
#sys.path.insert(0, '/afs/cern.ch/user/l/lian/public/diHiggs/dihiggsMLProject/higgsReconstruction')
sys.path.insert(0,'../')

from eventReconstructionClass_Jet import eventReconstruction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input',dest='filename')
parser.add_argument('--diHiggs',dest='diHiggs')
parser.add_argument('--inputTag',dest='tag')

args = parser.parse_args()

process_idx = 0
if(args.diHiggs=='False'):
  diHiggs = False
elif(args.diHiggs=='True'):
  diHiggs = True
else:
  raise ValueError("Invalid input")

with open(args.filename) as f:
  line = f.readline()
  while line:
    file_to_process = line.strip('\n')
    print(args.diHiggs)
    reco = eventReconstruction(args.tag, file_to_process, diHiggs, _isTestRun = False, _saveIdx=process_idx)
    reco.setConsiderFirstNjetsInPT(4)
    reco.setNJetsToStore(10)
    reco.setRequireNTags(4)
    reco.setJetPtEtaCuts(20,2.0)
    reco.runReconstruction()
    process_idx += 1
    line = f.readline()

