import csv
import subprocess
import ROOT as R
import math as M

sig_name = 'dihiggs_outputDataForLearning.csv'
bkg_name = 'qcd_outputDataForLearning.csv'
f_name = sig_name
l = int(subprocess.check_output(["wc","-l",f_name]).split()[0])
vars_list = ['hh_mass', 'h1_mass', 'h2_mass', 'hh_pt', 'h1_pt', 'h2_pt', 'deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)', 'deltaPhi(h1, h2)', 'deltaPhi(h1 jets)', 'deltaPhi(h2 jets)', 'met', 'met_phi', 'scalarHT', 'nJets', 'nBTags', 'jet1_pt', 'jet2_pt', 'jet3_pt', 'jet4_pt', 'jet1_eta', 'jet2_eta', 'jet3_eta', 'jet4_eta', 'jet1_phi', 'jet2_phi', 'jet3_phi', 'jet4_phi', 'jet1_mass', 'jet2_mass', 'jet3_mass', 'jet4_mass', 'jet1_px', 'jet2_px', 'jet3_px', 'jet4_px', 'jet1_py', 'jet2_py', 'jet3_py', 'jet4_py', 'jet1_pz', 'jet2_pz', 'jet3_pz', 'jet4_pz', 'jet1_energy', 'jet2_energy', 'jet3_energy', 'jet4_energy']
var_namel = ['hh_mass', 'h1_mass', 'h2_mass', 'hh_pt', 'h1_pt', 'h2_pt', 'deltaR_h1__h2', 'deltaR_h1_jet', 'deltaR_h2_jet', 'deltaPhi_h1__h2', 'deltaPhi_h1_jet', 'deltaPhi_h2_jet', 'met', 'met_phi', 'scalarHT', 'nJets', 'nBTags', 'jet1_pt', 'jet2_pt', 'jet3_pt', 'jet4_pt', 'jet1_eta', 'jet2_eta', 'jet3_eta', 'jet4_eta', 'jet1_phi', 'jet2_phi', 'jet3_phi', 'jet4_phi', 'jet1_mass', 'jet2_mass', 'jet3_mass', 'jet4_mass', 'jet1_px', 'jet2_px', 'jet3_px', 'jet4_px', 'jet1_py', 'jet2_py', 'jet3_py', 'jet4_py', 'jet1_pz', 'jet2_pz', 'jet3_pz', 'jet4_pz', 'jet1_energy', 'jet2_energy', 'jet3_energy', 'jet4_energy']

histos = {}
histos["hh_mass"] = R.TH1F("hh_mass","hh_mass",400,0,1200)
histos["h1_mass"] = R.TH1F("h1_mass","h1_mass",200,0,600)
histos["h2_mass"] = R.TH1F("h2_mass","h2_mass",200,0,600)
histos["hh_pt"] = R.TH1F("hh_pt","hh_pt",250,0,500)
histos["h1_pt"] = R.TH1F("h1_pt","h1_pt",250,0,500)
histos["h2_pt"] = R.TH1F("h2_pt","h2_pt",250,0,500)
histos["deltaR_h1__h2"] = R.TH1F("deltaR_h1__h2","deltaR_h1__h2",200,0,5)
histos["deltaR_h1_jet"] = R.TH1F("deltaR_h1_jet","deltaR_h1_jet",200,0,5)
histos["deltaR_h2_jet"] = R.TH1F("deltaR_h2_jet","deltaR_h2_jet",200,0,5)
histos["deltaPhi_h1__h2"] = R.TH1F("deltaPhi_h1__h2","deltaPhi_h1__h2",200,-4,4)
histos["deltaPhi_h1_jet"] = R.TH1F("deltaPhi_h1_jet","deltaPhi_h1_jet",200,-4,4)
histos["deltaPhi_h2_jet"] = R.TH1F("deltaPhi_h2_jet","deltaPhi_h2_jet",200,-4,4)
histos["met"] = R.TH1F("met","met",200,0,200)
histos["met_phi"] = R.TH1F("met_phi","met_phi",800,-4,4)
histos["scalarHT"] = R.TH1F("scalarHT","scalarHT",1000,0,1000)
histos["nJets"] = R.TH1F("nJets","nJets",12,0,12)
histos["nBTags"] = R.TH1F("nBTags","nBTags",10,0,10)
jet_var = ['jet1_pt', 'jet2_pt', 'jet3_pt', 'jet4_pt', 'jet1_eta', 'jet2_eta', 'jet3_eta', 'jet4_eta', 'jet1_phi', 'jet2_phi', 'jet3_phi', 'jet4_phi', 'jet1_mass', 'jet2_mass', 'jet3_mass', 'jet4_mass', 'jet1_px', 'jet2_px', 'jet3_px', 'jet4_px', 'jet1_py', 'jet2_py', 'jet3_py', 'jet4_py', 'jet1_pz', 'jet2_pz', 'jet3_pz', 'jet4_pz', 'jet1_energy', 'jet2_energy', 'jet3_energy', 'jet4_energy']
for ij in jet_var:
    j_range_up = float(0)
    j_range_down = float(0)
    j_bin = int(0)
    if (ij.find('pt')>0):
        j_range_up = 500
        j_range_down = 0
        j_bin = 500
    elif (ij.find('eta')>0):
        j_range_up = 3
        j_range_down = -3
        j_bin = 600
    elif (ij.find('phi')>0):
        j_range_up = 4
        j_range_down = -4
        j_bin = 800
    elif (ij.find('mass')>0):
        j_range_up = 50
        j_range_down = 0
        j_bin = 500
    elif (ij.find('px')>0):
        j_range_up = 300
        j_range_down = -300
        j_bin = 6000
    elif (ij.find('py')>0):
        j_range_up = 300
        j_range_down = -300
        j_bin = 6000
    elif (ij.find('pz')>0):
        j_range_up = 1000
        j_range_down = -1000
        j_bin = 10000
    elif (ij.find('energy')>0):
        j_range_up = 500
        j_range_down = 0
        j_bin = 500
    histos[ij] = R.TH1F(ij,ij,j_bin,j_range_down,j_range_up)




with open(f_name) as f:
    r = csv.DictReader(f)
    for nr, row in enumerate(r):
        if(nr<=l/2):
            continue
        #if(nr>l/2):   
        #    break
        for i in range(0,len(vars_list)):
            histos[var_namel[i]].Fill(float(row[vars_list[i]]))
            if(nr==1):
                print (vars_list[i])
                print (var_namel[i])
                print (float(row[vars_list[i]]))
        



output = 'pdf_sig.root'
fOut=R.TFile(output,"RECREATE")
for hn, histo in histos.iteritems():
    histo.Scale(1.0/(histo.Integral()))
    histo.Write()
fOut.Close()
print ("Saved histos in "+output)



