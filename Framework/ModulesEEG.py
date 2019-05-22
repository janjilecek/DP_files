from scipy.integrate import simps
from scipy.signal import welch
import numpy as np
import pywt
import pywt.data
from collections import Counter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.io

final_data = []
final_labels = []


class EEGAnalysis:
    def __init__(self, fn="./data/left_hand_CLOSED_LeftH_12_ironman_DOMINIKA_1555532165.json"):
        self.fn = fn
        self.arr = []
        self.numpied = None
        self.finalDict = dict()
        self.wholeArr = None
        self.data_left = []
        self.data_right = []
        self.overlap = 50
        self.finalLabels = []


    def extractCsvChannelIntoArray(self, channel='fp1'):
        print("loading")
        try:
            data= pd.read_csv(self.fn, delimiter=',')
        except Exception as e:
            print(e)
        print("loaded")
        try:
            print(data)
            a = data['fp1']  # test if is indexable; if not, it is not comma delimited
        except:
            data = pd.read_csv(self.fn, delimiter=';')
            print(data)
            pass
        print("fal")
        self.wholeArr = data
        self.numpied = data[channel]

    def transformInputData(self, sampleRate, dropData=True, startDrop=0, endDrop=0):
        self.data_right = self.wholeArr.loc[self.wholeArr['class'] == 2]
        self.data_left = self.wholeArr.loc[self.wholeArr['class'] == 1]
        trimmer = 0.0

        if dropData:
            secondsToDrop = 2  # drop first X seconds in the CHALLENGE
            for i, now in enumerate(self.data_right['timestamp']):
                diff = now - trimmer
                trimmer = now
                if diff > 1.0:
                    self.data_right = self.data_right.drop(
                        self.data_right.index[i - endDrop * sampleRate: i + startDrop * sampleRate])
            for i, now in enumerate(self.data_left['timestamp']):
                diff = now - trimmer
                trimmer = now
                if diff > 1.0:  # if the time difference is bigger than 1 second, it is another challenge
                    self.data_left = self.data_left.drop(self.data_left.index[i - endDrop * sampleRate: i + startDrop * sampleRate])

    def wavelet_decomposition(self, arr, name="..", wvlt='sym9', write=True):
        wavelet = pywt.Wavelet(wvlt)
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))  # to normalize the data between 0 and 1.0
        mode = pywt.Modes.smooth  # unused now
        approximations = []
        details = []
        data = a = arr
        for i in range(6):
            (a, d) = pywt.dwt(a, wavelet)  # removed mode, jde to nahoru pak
            approximations.append(a)
            details.append(d)

        rec_details = []
        rec_approx = []


        acka = ""
        for i, coeff in enumerate(approximations):
            coeff_list = [coeff, None] + [None] * i
            fitted = scaler.fit_transform(pywt.waverec(coeff_list, wavelet).reshape(-1, 1))  # scale data
            a = []
            for b in fitted[0:250]:  # bad format, let's adapt it
                a.extend(b)

            acka += str(len(a))+";"
            rec_approx.append(a)

        for i, coeff in enumerate(details):
            coeff_list = [None, coeff] + [None] * i
            fitted = scaler.fit_transform(pywt.waverec(coeff_list, wavelet).reshape(-1, 1))
            a = []
            for b in fitted[0:250]:
                a.extend(b)
            acka += str(len(a)) + ";"
            rec_details.append(a)
        if write:
            with open("stats_wavs2.csv", "a") as f:
                f.write(wvlt+ ";"+name+";" + acka + '\n')

        return (rec_approx, rec_details)  # return tuple

    # UNUSED
    def power_spectral_density(self, arr):
        N = len(self.numpied)  # samples
        t_n = N / 250  # seconds
        T = t_n / N
        f_s = 250.  # sampling frequency
        f_values, psd_values = welch(arr, fs=f_s, window='hamming', nperseg=250)
        low, high = 7, 15
        idx_delta = np.logical_and(f_values >= low, f_values <= high)
        freq_res = f_values[1] - f_values[0]  # = 1 / 4 = 0.25
        delta_power = simps(psd_values[idx_delta], dx=freq_res)

        return psd_values[idx_delta]

    # UNUSED
    def fft_on_series(self, arr):
        fft_vals = np.absolute(np.fft.rfft(arr))
        fft_freq = np.fft.rfftfreq(arr.size, 1.0 / 250.)
        return fft_vals[:48]

    def calculate_entropy(self, list_values):
        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1] / len(list_values) for elem in counter_values]
        entropy = scipy.stats.entropy(probabilities)
        return entropy

    def calculate_statistics(self, list_values):
        n5 = np.nanpercentile(list_values, 5)
        n25 = np.nanpercentile(list_values, 25)
        n75 = np.nanpercentile(list_values, 75)
        n95 = np.nanpercentile(list_values, 95)
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        expos = np.array(len(list_values) * [2])
        power =  np.average(np.power(list_values, expos))
        return [median, mean, std, var, power]


    def calculate_crossings(self, list_values):
        zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        return [no_zero_crossings, no_mean_crossings]

    def get_features(self, list_values):
        entropy = self.calculate_entropy(list_values)
        crossings = self.calculate_crossings(list_values)
        statistics = self.calculate_statistics(list_values)
        return [entropy] + crossings + statistics

    def get_all_feats(self, arr, start=0, end=5):
        miarr = []
        for a in arr[start:end]:
            miarr.extend(a)
        return miarr

    def compute_class(self, classNumber, hand_data, _wavelet, lesserApprox, upperApprox, lesserDetails, upperDetails,
                      doStats, plot_psd, wholeArrC3, wholeArrC4, wholeArrCz,
                      c3_d, c4_d, cz_d, _fft, c3_feat, c4_feat, cz_feat, wantApprox, _samples=250, _overlap=250,
                      electrodes=[True, False, True]):
        samples = _samples
        overlap = _overlap

        index = 0
        C3_ON = electrodes[0]
        CZ_ON = electrodes[1]
        C4_ON = electrodes[2]

        while index < len(hand_data):
            try:
                # first process the left side, then the right
                c3_arr = hand_data['c3'][index:index + samples]
                c4_arr = hand_data['c4'][index:index + samples]

                try:
                    cz_arr = hand_data['cz'][index:index + samples]
                    z = 5
                except Exception as e:
                    print(e)
                    print("Turning off CZ electrode")
                    CZ_ON = False

                if plot_psd:
                    if C3_ON:
                        wholeArrC3.append(self.power_spectral_density(c3_arr))  # c3

                    if C4_ON:
                        wholeArrC4.append(self.power_spectral_density(c4_arr))

                    if CZ_ON:
                        wholeArrCz.append(self.power_spectral_density(cz_arr))
                else:
                    if C3_ON:
                        c3_wavelets_hand = self.wavelet_decomposition(c3_arr, wvlt=_wavelet, name="leftC3", write=index < 2)
                    if C4_ON:
                        c4_wavelets_hand = self.wavelet_decomposition(c4_arr, wvlt=_wavelet, name="leftC4", write=index < 2)
                    if CZ_ON:
                        cz_wavelets_hand = self.wavelet_decomposition(cz_arr, wvlt=_wavelet, name="leftCz", write=index < 2)

                    if doStats:
                        if C3_ON:
                            if wantApprox:
                                wholeArrC3.append(self.get_all_feats(c3_wavelets_hand[0], lesserApprox, upperApprox))
                            c3_d.append(self.get_all_feats(c3_wavelets_hand[1], lesserDetails, upperDetails))
                        if C4_ON:
                            if wantApprox:
                                wholeArrC4.append(self.get_all_feats(c4_wavelets_hand[0], lesserApprox, upperApprox))
                            c4_d.append(self.get_all_feats(c4_wavelets_hand[1], lesserDetails, upperDetails))
                        if CZ_ON:
                            if wantApprox:
                                wholeArrCz.append(self.get_all_feats(cz_wavelets_hand[0], lesserApprox, upperApprox))
                            cz_d.append(self.get_all_feats(cz_wavelets_hand[1], lesserDetails, upperDetails))



                    else:  # doing feautures or fft - UNUSED
                        if _fft:  # get simple fft
                            if C3_ON:
                                fft3 = self.fft_on_series(c3_arr)
                                c3_feat.append(fft3)
                            if C4_ON:
                                fft4 = self.fft_on_series(c4_arr)
                                c4_feat.append(fft4)
                            if CZ_ON:
                                fftz = self.fft_on_series(cz_arr)
                                cz_feat.append(fftz)
                        else:  # get only features
                            if C3_ON:
                                arrTemp = []
                                for wav_details in c3_wavelets_hand[1][lesserDetails:upperDetails]:
                                    arrTemp.append(self.get_features(wav_details))

                                if wantApprox:
                                    for wav_details in c3_wavelets_hand[0][lesserApprox:upperApprox]:
                                        arrTemp.append(self.get_features(wav_details))

                                c3_feat.append(self.get_all_feats(arrTemp, end=-1))

                            if C4_ON:
                                arrTemp = []
                                for wav_details in c4_wavelets_hand[1][lesserDetails:upperDetails]:
                                    arrTemp.append(self.get_features(wav_details))
                                if wantApprox:
                                    for wav_details in c4_wavelets_hand[0][lesserApprox:upperApprox]:
                                        arrTemp.append(self.get_features(wav_details))
                                c4_feat.append(self.get_all_feats(arrTemp, end=-1))

                            if CZ_ON:
                                arrTemp = []
                                for wav_details in cz_wavelets_hand[1][lesserDetails:upperDetails]:
                                    arrTemp.append(self.get_features(wav_details))
                                if wantApprox:
                                    for wav_details in cz_wavelets_hand[0][lesserApprox:upperApprox]:
                                        arrTemp.append(self.get_features(wav_details))
                                cz_feat.append(self.get_all_feats(arrTemp, end=-1))

                self.finalLabels.append(classNumber)
            except Exception as e:
                print(e)
                break
            index += overlap
        print("Hand data loaded;")
        return final_labels
