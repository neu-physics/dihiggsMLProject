import csv
import subprocess
import ROOT as R
import math as M
import sys
sys.path.insert(0, '/Users/flywire/Desktop/sci/dihiggsMLProject/')
from utils.commonFunctions import *

def makeLL(histo,var,var_h,name):
    histos["LL_s_s"+name] = R.TH1F("LL_s_s"+name,"LL_s_s"+name,2010,-200,10)
    histos["LL_s_b"+name] = R.TH1F("LL_s_b"+name,"LL_s_b"+name,2010,-200,10)
    histos["LL_b_s"+name] = R.TH1F("LL_b_s"+name,"LL_b_s"+name,2010,-200,10)
    histos["LL_b_b"+name] = R.TH1F("LL_b_b"+name,"LL_b_b"+name,2010,-200,10)
    histos["LLR_s"+name] = R.TH1F("LLR_s"+name,"LLR_s"+name,600,-60,60)
    histos["LLR_b"+name] = R.TH1F("LLR_b"+name,"LLR_b"+name,600,-60,60)
    histos["LL_2D_s"+name] = R.TH2F("LL_2D_s"+name,"LL_2D_s"+name,2010,-200,10,2010,-200,10)
    histos["LL_2D_b"+name] = R.TH2F("LL_2D_b"+name,"LL_2D_b"+name,2010,-200,10,2010,-200,10)

    testingFraction = 0.3
    lumiscale_hh  = getLumiScaleFactor(testingFraction, True)
    lumiscale_qcd = getLumiScaleFactor(testingFraction, False)
    
    sig_name = 'dihiggs_outputDataForLearning.csv'
    #bkg_name = 'qcd_outputDataForLearning.csv'
    bkg_name = 'qcd_2M_training.csv'
    sig_l = int(subprocess.check_output(["wc","-l",sig_name]).split()[0])
    bkg_l = int(subprocess.check_output(["wc","-l",bkg_name]).split()[0])

    pdf_sig = 'pdf_sig_500k_07.root'
    pdf_bkg = 'pdf_bkg_2M_07.root'
    #pdf_bkg = 'pdf_bkg.root'


    f_pdf_sig = R.TFile(pdf_sig)
    f_pdf_bkg = R.TFile(pdf_bkg)
    pdf_sig = {}
    pdf_bkg = {}
    for v in var_h:
        pdf_sig[v] = f_pdf_sig.Get(v)
        pdf_bkg[v] = f_pdf_bkg.Get(v)

    with open(sig_name) as f_sig:
        r_sig = csv.DictReader(f_sig)
        for nr, row in enumerate(r_sig):
            if(nr<=sig_l*(1-testingFraction)):
                continue
            ll_sig = 0
            ll_bkg = 0
            valid_s = True
            valid_b = True
            for i in range(0,len(var)):
                value = float(row[var[i]])
                #pdf_sig[v].Draw()
                v = var_h[i]
                nbin_sig = pdf_sig[v].FindBin(value)
                nbin_bkg = pdf_bkg[v].FindBin(value)
                prob_sig = pdf_sig[v].GetBinContent(nbin_sig)
                prob_bkg = pdf_bkg[v].GetBinContent(nbin_bkg)
                if ( valid_s and prob_sig>0 ):
                    ll_sig += M.log(prob_sig)
                else:
                    valid_s = False
                if ( valid_b and prob_bkg>0 ):
                    ll_bkg += M.log(prob_bkg)
                else:
                    valid_b = False

                #if ( prob_sig>0 and prob_bkg>0 ):
                #    ll_sig += M.log(prob_sig)
                #    ll_bkg += M.log(prob_bkg)
                #else:
                #    ll_sig = 1
                #    ll_bkg = 1
                #    break
            if (not valid_s):
                ll_sig = 1
            if (not valid_b):
                ll_bkg = 1
            histos["LL_s_s"+name].Fill(ll_sig)
            histos["LL_s_b"+name].Fill(ll_bkg)
            histos["LL_2D_s"+name].Fill(ll_sig,ll_bkg)
            if( (ll_sig<=0) and (ll_bkg<=0) ):
                histos["LLR_s"+name].Fill(ll_sig-ll_bkg)
            else:
                histos["LLR_s"+name].Fill(50)
    print(lumiscale_hh)
    histos["LL_s_s"+name].Scale(lumiscale_hh)
    histos["LL_s_b"+name].Scale(lumiscale_hh)
    histos["LL_2D_s"+name].Scale(lumiscale_hh)
    histos["LLR_s"+name].Scale(lumiscale_hh)



    with open(bkg_name,encoding='utf-8-sig') as f_bkg:
        r_bkg = csv.DictReader(f_bkg)
        for nr, row in enumerate(r_bkg):
            if(nr<=bkg_l*(1-testingFraction)):   
                continue
            ll_sig = 0
            ll_bkg = 0
            valid_s = True            
            valid_b = True
            for i in range(0,len(var)):
                value = float(row[var[i]])
                v = var_h[i]
                nbin_sig = pdf_sig[v].FindBin(value)
                nbin_bkg = pdf_bkg[v].FindBin(value)
                prob_sig = pdf_sig[v].GetBinContent(nbin_sig)
                prob_bkg = pdf_bkg[v].GetBinContent(nbin_bkg)
                if ( valid_s and prob_sig>0 ):
                    ll_sig += M.log(prob_sig)
                else:
                    valid_s = False
                    #print ('bkg on sig------var: {0} value: {1}'.format(v,value))
                if ( valid_b and prob_bkg>0 ):
                    ll_bkg += M.log(prob_bkg)
                else:
                    valid_b = False
                    #print ('bkg on bkg------var: {0} value: {1}'.format(v,value))

                
                #if ( prob_sig>0 and prob_bkg>0 ):
                #    ll_sig += M.log(prob_sig)
                #    ll_bkg += M.log(prob_bkg)
                #else:
                #    ll_sig = 1
                #    ll_bkg = 1
                #    break
            if (not valid_s):
                ll_sig = 1
            if (not valid_b):
                ll_bkg = 1
            histos["LL_b_s"+name].Fill(ll_sig)
            histos["LL_b_b"+name].Fill(ll_bkg)
            histos["LL_2D_b"+name].Fill(ll_sig,ll_bkg)
            if( (ll_sig<=0) and (ll_bkg<=0) ):
                histos["LLR_b"+name].Fill(ll_sig-ll_bkg)
            else:
                histos["LLR_b"+name].Fill(50)

    print(lumiscale_qcd)
    histos["LL_b_s"+name].Scale(lumiscale_qcd)
    histos["LL_b_b"+name].Scale(lumiscale_qcd)
    histos["LL_2D_b"+name].Scale(lumiscale_qcd)
    histos["LLR_b"+name].Scale(lumiscale_qcd)


histos = {}

var_mass = ['h1_mass','h2_mass','hh_mass']
var_mass_h = ['h1_mass','h2_mass','hh_mass']
#makeLL(histos,var_mass,var_mass_h,'_mass')

var_pt = ['hh_pt', 'h1_pt', 'h2_pt']
var_pt_h = ['hh_pt', 'h1_pt', 'h2_pt']
#makeLL(histos,var_pt,var_pt_h,'_pt')

var_dR = ['deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)']
var_dR_h = ['deltaR_h1__h2', 'deltaR_h1_jet', 'deltaR_h2_jet']
#makeLL(histos,var_dR,var_dR_h,'_dR')

var_all = ['hh_mass', 'h1_mass', 'h2_mass', 'hh_pt', 'h1_pt', 'h2_pt', 'deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)', 'deltaPhi(h1, h2)', 'deltaPhi(h1 jets)', 'deltaPhi(h2 jets)', 'met', 'met_phi', 'scalarHT', 'nJets', 'nBTags', 'jet1_pt', 'jet2_pt', 'jet3_pt', 'jet4_pt', 'jet1_eta', 'jet2_eta', 'jet3_eta', 'jet4_eta', 'jet1_phi', 'jet2_phi', 'jet3_phi', 'jet4_phi', 'jet1_mass', 'jet2_mass', 'jet3_mass', 'jet4_mass', 'jet1_px', 'jet2_px', 'jet3_px', 'jet4_px', 'jet1_py', 'jet2_py', 'jet3_py', 'jet4_py', 'jet1_pz', 'jet2_pz', 'jet3_pz', 'jet4_pz', 'jet1_energy', 'jet2_energy', 'jet3_energy', 'jet4_energy']
var_all_h = ['hh_mass', 'h1_mass', 'h2_mass', 'hh_pt', 'h1_pt', 'h2_pt', 'deltaR_h1__h2', 'deltaR_h1_jet', 'deltaR_h2_jet', 'deltaPhi_h1__h2', 'deltaPhi_h1_jet', 'deltaPhi_h2_jet', 'met', 'met_phi', 'scalarHT', 'nJets', 'nBTags', 'jet1_pt', 'jet2_pt', 'jet3_pt', 'jet4_pt', 'jet1_eta', 'jet2_eta', 'jet3_eta', 'jet4_eta', 'jet1_phi', 'jet2_phi', 'jet3_phi', 'jet4_phi', 'jet1_mass', 'jet2_mass', 'jet3_mass', 'jet4_mass', 'jet1_px', 'jet2_px', 'jet3_px', 'jet4_px', 'jet1_py', 'jet2_py', 'jet3_py', 'jet4_py', 'jet1_pz', 'jet2_pz', 'jet3_pz', 'jet4_pz', 'jet1_energy', 'jet2_energy', 'jet3_energy', 'jet4_energy']
#makeLL(histos,var_all,var_all_h,'_all')

var_part = ['h1_mass', 'h2_mass', 'deltaR(h1 jets)', 'deltaR(h2 jets)',  'deltaPhi(h1 jets)', 'deltaPhi(h2 jets)']
var_part_h = ['h1_mass', 'h2_mass', 'deltaR_h1_jet', 'deltaR_h2_jet', 'deltaPhi_h1_jet', 'deltaPhi_h2_jet']
makeLL(histos,var_part,var_part_h,'_part')


output = 'output_multi_QCD2M_07.root'
fOut=R.TFile(output,"RECREATE")
#for hn, histo in histos.iteritems():
for hn, histo in histos.items():
   histo.Write()
fOut.Close()
print ("Saved histos in "+output)