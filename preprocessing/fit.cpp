#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <algorithm>

const std::string IN_PATH = "/media/work/Waveforms/run4/100-200.root";
const std::string OUT_PATH = "/media/work/Waveforms/run4/analysis/amplitudes/new/100-200-2.root";

const bool APPEND = false;

// The first channel to consider...
const int FIRST_CHANNEL = 0;
// ...and the last one (included)
const int LAST_CHANNEL = 10;

// How many samples to take before...
const int BEFORE_PEAK = 3;
// ...and after peak to do a gaussian fit
const int AFTER_PEAK = 3;

// How many samples to use to calculate baseline, starting from the end
const int NUM_AVERAGE = 100;
// Reciprocal of frequency
const double TIME_STEP = 2E-10;
// Max value for fit chi2 to accept result
const double MAX_CHI2 = 1.5;

// Don't touch!
const int NUM_CHANNELS = 1 + LAST_CHANNEL - FIRST_CHANNEL;
const int NUM_SAMPLES = BEFORE_PEAK + AFTER_PEAK + 1;

template<typename T>
struct square {
    T operator()(const T& first, const T& second) const {
        return (first + second * second);
    }
};

double getRMS(std::vector<double>* samples) {
    return std::accumulate(samples->end() - NUM_AVERAGE,
        samples->end(), 0, square<double>()) / NUM_AVERAGE;
}

double getMean(std::vector<double>* samples) {
    return std::accumulate(samples->end() - NUM_AVERAGE,
        samples->end(), 0) / NUM_AVERAGE;
}

std::vector<double> getInterestZone(std::vector<double>* samples) {
    int peak = std::max_element(samples->begin(),
        samples->end()) - samples->begin();
    auto begin = samples->begin() + peak - BEFORE_PEAK;
    auto end = samples->begin() + peak + AFTER_PEAK + 1;
    std::vector<double> event(begin, end);

    return event;
}

void analyze(TTree *in, TTree *out) {
    // Output tree setup
    std::vector<double>* position = nullptr;
    if(APPEND) {
        out->SetBranchAddress("pos", &position);
    } else {
        out->Branch("pos", &position);
    }

    std::vector<double> amplitudes(NUM_CHANNELS, 0);
    for(int i = FIRST_CHANNEL; i <= LAST_CHANNEL; i++) {
        std::string amp = "amp" + std::to_string(i);
        if(APPEND) {
            out->SetBranchAddress(amp.c_str(), &(amplitudes[i - FIRST_CHANNEL]));
        } else {
            out->Branch(amp.c_str(), &(amplitudes[i - FIRST_CHANNEL]));
        }
    }

    // Input tree setup
    in->SetBranchAddress("pos", &position);

    std::vector<std::vector<double>*> channels(NUM_CHANNELS, nullptr);
    for(int i = FIRST_CHANNEL; i <= LAST_CHANNEL; i++) {
        std::string chn = "w" + std::to_string(i);
        in->SetBranchAddress(chn.c_str(), &(channels[i - FIRST_CHANNEL]));
    }

    std::vector<double> time;
    for(int i = 0; i < NUM_SAMPLES; i++) {
        time.push_back(i * TIME_STEP);
    }

    TF1 fit("fit", "gaus");
    fit.SetParameter(1, NUM_SAMPLES * TIME_STEP / 2);
    fit.SetParLimits(1, 0, NUM_SAMPLES * TIME_STEP);

    int events = in->GetEntries();
    // Branches interation
    for(int i = 0; i < events; i++) {
        in->GetEntry(i);

        for(int j = FIRST_CHANNEL; j <= LAST_CHANNEL; j++) {
            std::vector<double>* channel = channels[j];
            double average = getMean(channel);

            std::vector<double> event = getInterestZone(channel);

            TGraph graph(NUM_SAMPLES, &time[0], &event[0]);
            fit.SetParameter(0, average);
            fit.SetParLimits(0, average, average + 2000);
            gSystem->RedirectOutput("/dev/null");
            graph.Fit("fit", "NMQS");
            gSystem->RedirectOutput(0, 0);

            amplitudes[j - FIRST_CHANNEL] = fit.GetParameter(0) - average;
            if(fit.GetParameter(0) - average > 2000 || fit.GetParameter(0) < 0) {
                std::cout << fit.GetParameter(0) << std::endl;
                return;
            }
        }

        out->Fill();
        if(i % 100 == 0) {
            std::cout << "Progress: " << i << "/" << events << std::endl;
            out->Write();
        }
    }

    out->Write();
    out->ResetBranchAddresses();
    in->ResetBranchAddresses();
}

void preprocessing() {
    TFile *in = TFile::Open(IN_PATH.c_str(), "READ");
    TTree *wfm = (TTree*) in->Get("wfm");

    TFile *out; TTree *data;
    if(APPEND) {
        out = TFile::Open(OUT_PATH.c_str(), "UPDATE");
        data = (TTree*) out->Get("data");
    } else {
        out = TFile::Open(OUT_PATH.c_str(), "CREATE");
        data = new TTree("data", "Analysis data");
    }

    analyze(wfm, data);

    in->Close();
    out->Close();
}
