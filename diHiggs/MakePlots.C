void significance(TH1F *s, TH1F *b, float up, float low, TString tag)
{
    const double L_s = 33034.175242;
    const double L_b = 1.357924;
    double w = L_b/L_s;
    int up_bin_s = s->FindBin(up);
    int up_bin_b = b->FindBin(up);
    int low_bin_s = s->FindBin(low);
    int low_bin_b = b->FindBin(low);
    double N_sig = 0;
    double N_bkg = 0;
    double sig_max = 0;
    vector<double> signi;
    vector<double> LL;
    for (int i = up_bin_s; i>=low_bin_s; --i){
        double S = s->Integral(i,up_bin_s);
        double B = b->Integral(i,up_bin_s);

        double sig = 0;
        if (B!=0){
            sig = (3000.*S*w)/sqrt(3000.*B);
            //std::cout << S << " " << B << " " << sig << std::endl;
        }
        double ll = s->GetBinCenter(i);
        signi.push_back(sig);
        LL.push_back(ll);
        if(sig > sig_max){
            sig_max = sig;
            N_sig = 3000.*S*w;
            N_bkg = 3000.*B;
        }
    }
    std::cout << "For " << tag << " significance max: " << sig_max << " N_sig: " << N_sig << " N_bkg: " << N_bkg << std::endl;
    TGraph *g = new TGraph(signi.size(),&LL[0],&signi[0]);
    TCanvas *c = new TCanvas("c","c",800,600);
    c->cd();
    g->SetTitle("significance of "+tag+";log-likelihood;significance");
    g->SetLineWidth(3);
    g->Draw("AL");
    c->Print(tag+"_significance.png");
}






void Add(TString name)
{
    const double L_s = 33034.175242;
    const double L_b = 1.357924;
    TString f_name = "output_multi_test.root";
    TFile *f = new TFile(f_name);
    TString p_name[6] = {"LL_s_s"+name, "LL_b_s"+name, "LL_s_b"+name, "LL_b_b"+name, "LLR_s"+name, "LLR_b"+name};
    TH1F *h[6];
    for (int i = 0; i<6; ++i){
        h[i] = (TH1F*)f->Get(p_name[i]);
        //h[i]->Scale(1.0/(h[i]->Integral()));
        h[i]->SetLineWidth(3);
        if(i%2){
            h[i]->SetLineColor(kRed);
        }
    }
    for (int n = 0; n<6; n+=2){
        h[n]->Scale((1.0*(h[n+1]->Integral()))/h[n]->Integral());
    }
    for(int k = 0; k<6; ++k){
        std::cout << h[k]->Integral() << std::endl;
    }

    TH2F *h2D_s = (TH2F*)f->Get("LL_2D_s"+name);
    TH2F *h2D_b = (TH2F*)f->Get("LL_2D_b"+name);

    significance(h[0],h[1],-20.0,-50.0,"LL against sig");
    significance(h[2],h[3],-20.0,-50.0,"LL against bkg");
    significance(h[4],h[5],15.0,-15.0,"LLR");

    TCanvas *c1 = new TCanvas("c1","c1",800,600);
    c1->cd();
    h[0]->SetTitle("LL using signal as PDF;LL;franction of entries");
    h[0]->GetXaxis()->SetRangeUser(-50,-20);
    h[0]->Draw("hist");
    h[1]->Draw("hist same");
    TLegend *l1 = new TLegend(0.1,0.9,0.35,0.75);
    l1->AddEntry(h[0],"signal");
    l1->AddEntry(h[1],"background");
    l1->Draw();
    c1->Print("LL_s"+name+".png");


    TCanvas *c2 = new TCanvas("c2","c2",800,600);
    c2->cd();
    h[2]->SetTitle("LL using background as PDF;LL;franction of entries");
    h[2]->GetXaxis()->SetRangeUser(-50,-20);
    h[2]->Draw("hist");
    h[3]->Draw("hist same");
    TLegend *l2 = new TLegend(0.1,0.9,0.35,0.75);
    l2->AddEntry(h[2],"signal");
    l2->AddEntry(h[3],"background");
    l2->Draw();
    c2->Print("LL_b"+name+".png");

    TCanvas *c3 = new TCanvas("c3","c3",800,600);
    c3->cd();
    h[4]->SetTitle("LLR;LLR;fraction of entries");
    h[4]->GetXaxis()->SetRangeUser(-15,15);
    h[4]->Draw("hist");
    h[5]->Draw("hist same");
    TLegend *l3 = new TLegend(0.1,0.9,0.35,0.75);
    l3->AddEntry(h[4],"signal");
    l3->AddEntry(h[5],"background");
    l3->Draw();
    c3->Print("LLR_b"+name+".png");


    TCanvas *c4 = new TCanvas("c4","c4",800,600);
    c4->cd();
    h2D_s->SetLineWidth(3);
    h2D_s->SetTitle("2D LL (signal samples);LL against signal;LL against bkg");
    h2D_s->GetXaxis()->SetRangeUser(-50,-20);
    h2D_s->GetYaxis()->SetRangeUser(-50,-20);
    h2D_s->Draw("colz");
    c4->Print("LL2D_s.png");

    TCanvas *c5 = new TCanvas("c5","c5",800,600);
    c5->cd();
    h2D_b->SetLineWidth(3);
    h2D_b->SetTitle("2D LL (bkg samples);LL against signal;LL against bkg");
    h2D_b->GetXaxis()->SetRangeUser(-50,-20);
    h2D_b->GetYaxis()->SetRangeUser(-50,-20);
    h2D_b->Draw("colz");
    c5->Print("LL2D_b.png");




    //f->Close();
}




void MakePlots()
{
    gStyle->SetOptStat(0);
    Add("_part");

}