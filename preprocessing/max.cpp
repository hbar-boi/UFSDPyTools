#include <vector>
#include <string>
#include <iostream>
#include <cstring>

int TRIGGER_PEAK_THRESH = 1400;

int getTriggerPeak(std::vector<double> *trigger) {
    int closest = 0;
    for(int i = 0; i < 1024; i++) {
        int delta = trigger->at(i) - TRIGGER_PEAK_THRESH;
        if(delta < trigger->at(closest) - TRIGGER_PEAK_THRESH) {
            closest = i;
        }
    }

    return closest;
}

int SAMPLES_PER_EVENT = 1024;

int EVENTS_PER_POINT = 100;
int AVERAGE_SAMPLES_BACK = 100;
int AVERAGE_SAMPLES_FRONT = 20;

// Increase this if error on vector::_M_fill_insert
int TRIGGER_MAX_SHIFT = 40;

int WINDOW_START = 125;
int WINDOW_END = 375;
int WINDOW_SIZE = WINDOW_END - WINDOW_START;

std::vector<double> getPeak(TTree* tree, int chn, int stride) {
    std::vector<double> *channel = 0;
    std::string chan = "w" + std::to_string(chn);
    tree->SetBranchAddress(chan.c_str(), &channel);

    std::vector<double> *trigger = 0;
    int trg = chn < 8 ? 0 : 1;
    std::string trig = "trg" + std::to_string(trg);
    tree->SetBranchAddress(trig.c_str(), &trigger);

    std::vector<double> *pos = 0;
    tree->SetBranchAddress("pos", &pos);

    std::vector<double> res(3, 0);

    std::vector<double> average(WINDOW_SIZE, 0);
    int firstTrigger = 0;
    for(int k = 0; k < EVENTS_PER_POINT; k++) {
        int delta;
        tree->GetEntry(stride + k);
        if(k == 0) {
            res[0] = pos->at(0);
            res[1] = pos->at(1);

            firstTrigger = getTriggerPeak(trigger);
            delta = TRIGGER_MAX_SHIFT;
        } else {
            delta = getTriggerPeak(trigger) - firstTrigger + TRIGGER_MAX_SHIFT;
        }

        double mean = std::accumulate(channel->end() - AVERAGE_SAMPLES_BACK,
            channel->end(), 0) / AVERAGE_SAMPLES_BACK;

        channel->insert(channel->end(), delta, mean);
        channel->erase(channel->begin(), channel->begin() + delta);

        for(int l = 0; l < WINDOW_SIZE; l++) {
            average[l] += channel->at(WINDOW_START + l);
        }
    }
    tree->ResetBranchAddresses();

    for(int l = 0; l < average.size(); l++) {
        average[l] /= SAMPLES_PER_EVENT;
    }

    double mean = std::accumulate(average.begin(),
        average.begin() + AVERAGE_SAMPLES_FRONT, 0) / AVERAGE_SAMPLES_FRONT;
    double top = 0;
    for(int l = 0; l < average.size(); l++) {
        average[l] -= mean;
        double current = average[l];
        if(current > top) {
            top = current;
        }
    }

    res[2] = top;
    return res;
}

void work(int which) {
    TFile *in = TFile::Open("/media/work/Waveforms/run4/100-200.root", "READ");
    TTree *tree = (TTree*) in->Get("wfm");

    int events = tree->GetEntries();
    int points = (int) events / EVENTS_PER_POINT;

    std::string path = "/media/work/Waveforms/run4/analysis/chn" + std::to_string(which) + "/heatmap.root";

    TFile *out = TFile::Open(path.c_str(), "CREATE");
    TTree *data = new TTree("heat", "Heatmap data");

    std::vector<double> pos(2, 0);
    double map = 0;

    data->Branch("pos", &pos);
    data->Branch("heat", &map);

    std::vector<double> res;
    for(int i = 0; i < points; i++) {
        std::cout << "Progress: " << i << "/" << points << std::endl;
        res = getPeak(tree, which, i * EVENTS_PER_POINT);

        pos[0] = res[0];
        pos[1] = res[1];

        map = res[2];
        data->Fill();

        if(i % 100 == 0) {
            data->Write();
        }
    }
    data->Write();
    out->Close();

    in->Close();
}

void preprocessing() {
    for(int p = 9; p < 10; p++) {
        work(p);
    }
}
