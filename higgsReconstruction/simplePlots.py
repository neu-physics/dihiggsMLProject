import uproot
import matplotlib.pyplot as plt
import numpy as np

t = 0.7  # transparency of plots

def fill_histograms_new(ht_4b, ht_hh4b, met_4b, met_hh4b, jetPt_4b, jetPt_hh4b):
    print("filling...")

    t_ht_4b = uproot.tree.TBranchMethods.array(ht_4b["ScalarHT.HT"])
    t_ht_hh4b = uproot.tree.TBranchMethods.array(ht_hh4b["ScalarHT.HT"])
    t_met_4b = uproot.tree.TBranchMethods.array(met_4b["MissingET.MET"])
    t_met_hh4b = uproot.tree.TBranchMethods.array(met_hh4b["MissingET.MET"])
    t_jetPt_4b = uproot.tree.TBranchMethods.array(jetPt_4b["Jet.PT"])
    t_jetPt_hh4b = uproot.tree.TBranchMethods.array(jetPt_hh4b["Jet.PT"])

    #plot_scalarht_new(t_ht_4b, t_ht_hh4b)
    plot_twoSamples(t_met_4b, t_met_hh4b, 'Missing ET At 0PU', 'Missing ET [GeV]', 1, 0, 200, 125)
    plot_twoSamples(t_ht_4b, t_ht_hh4b, 'Scalar HT At 0PU', 'Scalar HT [GeV]', 2, 0, 1000, 100)
    plot_twoSamples(t_jetPt_4b, t_jetPt_hh4b, 'All Jet PT At 0PU', 'Jet PT [GeV]', 3, 0, 250, 100)


def plot_twoSamples(h_4b, h_hh4b, title, xtitle, nPlot, xMin, xMax, nBins):

    h_4b = np.concatenate(h_4b).ravel().tolist()
    h_hh4b = np.concatenate(h_hh4b).ravel().tolist()

    mean_4b = np.mean(h_4b)
    stdev_4b = np.std(h_4b)
    nEntries_4b = len(h_4b)

    mean_hh4b = np.mean(h_hh4b)
    stdev_hh4b = np.std(h_hh4b)
    nEntries_hh4b = len(h_hh4b)


    s1 = "QCD 4b Production:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(mean_4b, mean_4b, stdev_4b)
    s2 = "Dihiggs > 4b Production:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(mean_hh4b, mean_hh4b, stdev_hh4b)
    s20 = s1 + "\n" + s2 + "\n"

    plt.figure(nPlot)
    plt.title(title)
    plt.xlabel(xtitle)
    bins = np.linspace(xMin, xMax, nBins)
    plt.hist(h_4b, bins, alpha=t, label='QCD 4b')
    plt.hist(h_hh4b, bins, alpha=t, label='hh > 4b')
    plt.legend(loc='upper right')
    #plt.text(100, 1000, s20)
    plt.show()



def fill_histograms(h20s, h200s, d20s, d200s, w20s, w200s,
                    h20j, h200j, d20j, d200j, w20j, w200j,
                    h20m, h200m, d20m, d200m, w20m, w200m):

    print("filling...")

    hza20scht = uproot.tree.TBranchMethods.array(h20s["ScalarHT.HT"])
    hza20jpt = uproot.tree.TBranchMethods.array(h20j["Jet.PT"])
    hza20met = uproot.tree.TBranchMethods.array(h20m["MissingET.MET"])

    hza200scht = uproot.tree.TBranchMethods.array(h200s["ScalarHT.HT"])
    hza200jpt = uproot.tree.TBranchMethods.array(h200j["Jet.PT"])
    hza200met = uproot.tree.TBranchMethods.array(h200m["MissingET.MET"])

    dy20scht = uproot.tree.TBranchMethods.array(d20s["ScalarHT.HT"])
    dy20jpt = uproot.tree.TBranchMethods.array(d20j["Jet.PT"])
    dy20met = uproot.tree.TBranchMethods.array(d20m["MissingET.MET"])

    dy200scht = uproot.tree.TBranchMethods.array(d200s["ScalarHT.HT"])
    dy200jpt = uproot.tree.TBranchMethods.array(d200j["Jet.PT"])
    dy200met = uproot.tree.TBranchMethods.array(d200m["MissingET.MET"])

    wb20scht = uproot.tree.TBranchMethods.array(w20s["ScalarHT.HT"])
    wb20jpt = uproot.tree.TBranchMethods.array(w20j["Jet.PT"])
    wb20met = uproot.tree.TBranchMethods.array(w20m["MissingET.MET"])

    wb200scht = uproot.tree.TBranchMethods.array(w200s["ScalarHT.HT"])
    wb200jpt = uproot.tree.TBranchMethods.array(w200j["Jet.PT"])
    wb200met = uproot.tree.TBranchMethods.array(w200m["MissingET.MET"])


    plot_scalarht(hza20scht, hza200scht, dy20scht, dy200scht, wb20scht, wb200scht)
    plot_jpt(hza20jpt, hza200jpt, dy20jpt, dy200jpt, wb20jpt, wb200jpt)
    plot_met(hza20met, hza200met, dy20met, dy200met, wb20met, wb200met)



def plot_jpt(h20, h200, d20, d200, w20, w200):

    h20 = np.concatenate(h20).ravel().tolist()
    d20 = np.concatenate(d20).ravel().tolist()
    w20 = np.concatenate(w20).ravel().tolist()
    h200 = np.concatenate(h200).ravel().tolist()
    d200 = np.concatenate(d200).ravel().tolist()
    w200 = np.concatenate(w200).ravel().tolist()

    meanh20 = np.mean(h20)
    stdh20 = np.std(h20)
    enth20 = len(h20)
    meanh200 = np.mean(h200)
    stdh200 = np.std(h200)
    enth200 = len(h200)

    meand20 = np.mean(d20)
    stdd20 = np.std(d20)
    entd20 = len(d20)
    meand200 = np.mean(d200)
    stdd200 = np.std(d200)
    entd200 = len(d200)

    meanw20 = np.mean(w20)
    stdw20 = np.std(w20)
    entw20 = len(w20)
    meanw200 = np.mean(w200)
    stdw200 = np.std(w200)
    entw200 = len(w200)

    s1 = "higgs z gamma:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(enth20, meanh20, stdh20)
    s2 = "drell yan:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(entd20, meand20, stdd20)
    s3 = "weak boson eft:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(entw20, meanw20, stdw20)
    s20 = s1 + "\n" + s2 + "\n" + s3

    s4 = "higgs z gamma:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(enth200, meanh200, stdh200)
    s5 = "drell yan:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(entd200, meand200, stdd200)
    s6 = "weak boson eft:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(entw200, meanw200, stdw200)
    s200 = s4 + "\n" + s5 + "\n" + s6

    plt.figure(3)
    plt.title("Jet PT For 20 PU")
    plt.xlabel("Jet PT")
    bins = np.linspace(0, 250, 100)
    plt.hist(h20, bins, alpha=t, label='higgs z gamma')
    plt.hist(d20, bins, alpha=t, label='drell yan')
    plt.hist(w20, bins, alpha=t, label='weak boson eft')
    plt.legend(loc='upper right')
    plt.text(100, 6000, s20)
    plt.show()

    plt.figure(4)
    plt.title("Jet PT For 200 PU")
    plt.xlabel("Jet PT")
    bins = np.linspace(0, 250, 100)
    plt.hist(h200, bins, alpha=t, label='higgs z gamma')
    plt.hist(d200, bins, alpha=t, label='drell yan')
    plt.hist(w200, bins, alpha=t, label='weak boson eft')
    plt.legend(loc='upper right')
    plt.text(100, 20000, s200)
    plt.show()

def plot_met(h20, h200, d20, d200, w20, w200):

    h20 = np.concatenate(h20).ravel().tolist()
    d20 = np.concatenate(d20).ravel().tolist()
    w20 = np.concatenate(w20).ravel().tolist()
    h200 = np.concatenate(h200).ravel().tolist()
    d200 = np.concatenate(d200).ravel().tolist()
    w200 = np.concatenate(w200).ravel().tolist()

    meanh20 = np.mean(h20)
    stdh20 = np.std(h20)
    enth20 = len(h20)
    meanh200 = np.mean(h200)
    stdh200 = np.std(h200)
    enth200 = len(h200)

    meand20 = np.mean(d20)
    stdd20 = np.std(d20)
    entd20 = len(d20)
    meand200 = np.mean(d200)
    stdd200 = np.std(d200)
    entd200 = len(d200)

    meanw20 = np.mean(w20)
    stdw20 = np.std(w20)
    entw20 = len(w20)
    meanw200 = np.mean(w200)
    stdw200 = np.std(w200)
    entw200 = len(w200)

    s1 = "higgs z gamma:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(enth20, meanh20, stdh20)
    s2 = "drell yan:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(entd20, meand20, stdd20)
    s3 = "weak boson eft:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(entw20, meanw20, stdw20)
    s20 = s1 + "\n" + s2 + "\n" + s3

    s4 = "higgs z gamma:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(enth200, meanh200, stdh200)
    s5 = "drell yan:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(entd200, meand200, stdd20)
    s6 = "weak boson eft:\nentries = {}, mean = {:.4F}, std dev = {:.4F}".format(entw200, meanw200, stdw200)
    s200 = s4 + "\n" + s5 + "\n" + s6

    plt.figure(5)
    plt.title("Missing ET For 20 PU")
    plt.xlabel("Missing ET")
    bins = np.linspace(0, 500, 100)
    plt.hist(h20, bins, alpha=t, label='higgs z gamma')
    plt.hist(d20, bins, alpha=t, label='drell yan')
    plt.hist(w20, bins, alpha=t, label='weak boson eft')
    plt.legend(loc='upper right')
    plt.text(100, 1000, s20)
    plt.show()

    plt.figure(6)
    plt.title("Missing ET For 200 PU")
    plt.xlabel("Missing ET")
    bins = np.linspace(0, 500, 100)
    plt.hist(h200, bins, alpha=t, label='higgs z gamma')
    plt.hist(d200, bins, alpha=t, label='drell yan')
    plt.hist(w200, bins, alpha=t, label='weak boson eft')
    plt.legend(loc='upper right')
    plt.text(200, 200, s200)
    plt.show()


def main():
    try:
        print("opening...")
        
        ht_ppTo4b_0PU = uproot.open('../../../MG5_aMC_v2_6_1/ppTo4b_14TeV/Events/run_02/tag_1_delphes_events.root')['Delphes']['ScalarHT']
        ht_ppToHHto4b_0PU = uproot.open('../../../MG5_aMC_v2_6_1/ppToHHto4b_14TeV/Events/run_02_decayed_1/tag_1_delphes_events.root')['Delphes']['ScalarHT']

        jetPt_ppTo4b_0PU = uproot.open('../../../MG5_aMC_v2_6_1/ppTo4b_14TeV/Events/run_02/tag_1_delphes_events.root')['Delphes']['Jet']
        jetPt_ppToHHto4b_0PU = uproot.open('../../../MG5_aMC_v2_6_1/ppToHHto4b_14TeV/Events/run_02_decayed_1/tag_1_delphes_events.root')['Delphes']['Jet']

        met_ppTo4b_0PU = uproot.open('../../../MG5_aMC_v2_6_1/ppTo4b_14TeV/Events/run_02/tag_1_delphes_events.root')['Delphes']['MissingET']
        met_ppToHHto4b_0PU = uproot.open('../../../MG5_aMC_v2_6_1/ppToHHto4b_14TeV/Events/run_02_decayed_1/tag_1_delphes_events.root')['Delphes']['MissingET']
        


    except FileNotFoundError:
        print("Incorrect file name - file not found")
        exit()


    fill_histograms_new(ht_ppTo4b_0PU, ht_ppToHHto4b_0PU,
                        met_ppTo4b_0PU, met_ppToHHto4b_0PU,
                        jetPt_ppTo4b_0PU, jetPt_ppToHHto4b_0PU    )


if __name__ == "__main__":
    main()
