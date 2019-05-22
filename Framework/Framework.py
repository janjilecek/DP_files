import os
import sys
import traceback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, 'software/python-3.4.1/gcc/lib/python3.4/site-packages/')  # pro METAVO doom server
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
from sklearn.utils import shuffle
import csv
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score
import time
from sklearn.metrics import roc_curve, auc
from ModulesEEG import EEGAnalysis

final_data = []
final_labels = []


def prepareFile(fn="clean_dataset\\1556726380_dataset_Martin.csv", test=False, wvlt='sym9', outName='train.csv',
                lesserApprox=3, upperApprox=6, lesserDetails=3, upperDetails=6,
                statsOn=True, plot_psd=False,
                _fft=False, wantApprox=False, _samples=250, _overlap=250,
                electrodes=[True, False, True], startDrop=1, endDrop=0):
    final_labels = []
    wholeArrC3 = []
    wholeArrC4 = []
    wholeArrCz = []
    c3_feat = []
    c4_feat = []
    cz_feat = []
    c3_d = []
    c4_d = []
    cz_d = []
    fileData = fn

    print("make dir")
    if not os.path.exists(wvlt):
        os.makedirs(wvlt)
    print("make dir: done")

    if fileData.endswith(".csv"):
        print("file ends with csv, yeah")
        eeg = EEGAnalysis(fileData)
        eeg.extractCsvChannelIntoArray('c3')  # no matter the channel, the array is done
        eeg.transformInputData(_samples, False, startDrop, endDrop)  # dont drop any data

        C3_ON = electrodes[0]
        CZ_ON = electrodes[1]
        C4_ON = electrodes[2]

        print("Lets go ona and compute the classes")
        eeg.compute_class(0, eeg.data_left, wvlt, lesserApprox, upperApprox, lesserDetails, upperDetails,
                          statsOn, plot_psd, wholeArrC3, wholeArrC4, wholeArrCz,
                          c3_d, c4_d, cz_d, _fft, c3_feat,
                          c4_feat, cz_feat, wantApprox, _samples, _overlap, [C3_ON, CZ_ON, C4_ON])
        eeg.compute_class(1, eeg.data_right, wvlt, lesserApprox, upperApprox, lesserDetails, upperDetails,
                          statsOn, plot_psd, wholeArrC3, wholeArrC4, wholeArrCz,
                          c3_d, c4_d, cz_d, _fft, c3_feat,
                          c4_feat, cz_feat, wantApprox, _samples, _overlap, [C3_ON, CZ_ON, C4_ON])

        print("saving from " + fileData)
        print("saving to " + outName)
        with open(outName, "w", newline='') as f:
            fieldnames = ['index', 'class']
            dataset_writer = csv.writer(f, delimiter=',')

            columnsWidth = 0
            rowsHeight = 0

            if statsOn:
                if C3_ON:
                    rowsHeight = len(c3_d)
                if C4_ON:
                    rowsHeight = len(c4_d)
                if CZ_ON:
                    rowsHeight = len(cz_d)

                if C3_ON:
                    if wantApprox:
                        columnsWidth += len(wholeArrC3[0])
                    columnsWidth += len(c3_d[0])
                if C4_ON:
                    if wantApprox:
                        columnsWidth += len(wholeArrC4[0])
                    columnsWidth += len(c4_d[0])
                if CZ_ON:
                    if wantApprox:
                        columnsWidth += len(wholeArrCz[0])
                    columnsWidth += len(cz_d[0])
            else:
                if C3_ON:
                    rowsHeight = len(c3_feat)
                    columnsWidth += len(c3_feat[0])
                if C4_ON:
                    rowsHeight = len(c4_feat)
                    columnsWidth += len(c4_feat[0])
                if CZ_ON:
                    rowsHeight = len(cz_feat)
                    columnsWidth += len(cz_feat[0])

            for i in range(0, columnsWidth):
                fieldnames.append(str(i))
            dataset_writer.writerow(fieldnames)

            i = 0

            csvIndex = 0

            avgIndex = 0
            for testAverage in range(0, rowsHeight):
                if C3_ON and not C4_ON:
                    avgIndex += len(c3_d[i]) if statsOn else len(c3_feat[i])
                if C4_ON and not C3_ON:
                    avgIndex += len(c4_d[i]) if statsOn else len(c4_feat[i])
                if C3_ON and C4_ON:
                    avgIndex += len(c4_d[i]) if statsOn else len(c4_feat[i])

            _avg = int(avgIndex / rowsHeight)

            while i < rowsHeight:
                label = eeg.finalLabels[i]
                if statsOn:
                    lenOfTheItems = 0
                    if C3_ON and C4_ON:
                        lenOfTheItems = len(c4_d[i]) + len(c3_d[i])
                    if C3_ON and not C4_ON:
                        lenOfTheItems = len(c3_d[i])
                    if C4_ON and not C3_ON:
                        lenOfTheItems = len(c4_d[i])

                    arry = [csvIndex, label]
                    csvIndex += 1

                    if C3_ON:
                        if wantApprox:
                            itemC3 = wholeArrC3[i]
                            arry.extend(itemC3)
                        arry.extend(c3_d[i])
                    if C4_ON:
                        if wantApprox:
                            itemC4 = wholeArrC4[i]
                            arry.extend(itemC4)
                        arry.extend(c4_d[i])
                    if CZ_ON:
                        if wantApprox:
                            itemCz = wholeArrCz[i]
                            arry.extend(itemCz)
                        arry.extend(cz_d[i])

                else:
                    arry = [csvIndex, label]
                    csvIndex += 1
                    if C3_ON:
                        arry.extend(c3_feat[i])
                    if C4_ON:
                        arry.extend(c4_feat[i])
                    if CZ_ON:
                        arry.extend(cz_feat[i])
                try:
                    dataset_writer.writerow(arry)
                except Exception as e:
                    print(e)
                    pass
                finally:
                    i += 1



def filter_data(arrin, electrodes, minAprox, maxAprox, minDet, maxDet, wantAprox, stats=False):
    print("STATS: " + str(stats))
    C3_ON = electrodes[0]
    CZ_ON = electrodes[1]
    C4_ON = electrodes[2]

    newsec = []
    for i in range(len(arrin)):
        second = arrin.iloc[[i]].values[0]
        arr = []
        if C3_ON:
            if wantAprox:

                if stats:
                    arr.extend(calculate_statistics(get_filered(second, minAprox, maxAprox, 0)))
                else:
                    arr.extend(get_filered(second, minAprox, maxAprox, 0))
            if stats:
                arr.extend(calculate_statistics(get_filered(second, minDet, maxDet, 1500)))
            else:
                arr.extend(get_filered(second, minDet, maxDet, 1500))
        if C4_ON:
            if wantAprox:

                if stats:
                    arr.extend(calculate_statistics(get_filered(second, minAprox, maxAprox, 3000)))
                else:
                    arr.extend(get_filered(second, minAprox, maxAprox, 3000))

            if stats:
                arr.extend(calculate_statistics(get_filered(second, minDet, maxDet, 4500)))
            else:
                arr.extend(get_filered(second, minDet, maxDet, 4500))

        if CZ_ON:
            if wantAprox:

                if stats:
                    arr.extend(calculate_statistics(get_filered(second, minAprox, maxAprox, 6000)))
                else:
                    arr.extend(get_filered(second, minAprox, maxAprox, 6000))

            if stats:
                arr.extend(calculate_statistics(get_filered(second, minDet, maxDet, 7500)))
            else:
                arr.extend(get_filered(second, minDet, maxDet, 7500))
        newsec.append(arr)

    return newsec


# paramter: input list with numerical values
# returns: 5 statistics
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    expos = np.array(len(list_values) * [2])
    power = np.average(np.power(list_values, expos))
    return [median, mean, std, var, power]

def get_filered(second, _min, _max, electrodeStart):
    ixt = [0, 250, 500, 750, 1000, 1250]
    ixs = [250, 500, 750, 1000, 1250, 1500]
    res = []

    for i, val in enumerate(ixt):
        if _min <= i <= _max:
            res.extend(second[ixt[i] + electrodeStart:ixs[i] + electrodeStart])
    return res


# prepares datasets - applies filtering, fills missing values, shuffles the datasets, and generally picks only the specified parameters
def get_datasets_ready(testedSecond, electrodes, minAppprox,
                       maxApprox, minDets, maxDets, wantApprox, wavelet,
                       doSeconds=False,
                       _stats=False,
                       _pca=False,
                       _ica=False,
                       _comp=10):
    print("Getting dataset ready" + str(time.time()))
    shafSeed = 48
    eegdata = pd.read_csv(wavelet + "/" + "train.csv")
    eegdata = eegdata.fillna(0)

    eegtest = pd.read_csv(wavelet + "/" + "test.csv")
    eegtest = eegtest.fillna(0)

    # test
    X_test = eegtest.drop('class', axis=1).drop('index', axis=1)
    y_test = eegtest['class']
    X_test = X_test[testedSecond::11] if doSeconds else X_test
    y_test = y_test[testedSecond::11] if doSeconds else y_test
    X_test = filter_data(X_test, electrodes, minAppprox, maxApprox, minDets, maxDets, wantApprox, stats=_stats)
    X_test, y_test = shuffle(X_test, y_test, random_state=shafSeed)

    # train
    X_train = eegdata.drop('class', axis=1).drop('index', axis=1)
    y_train = eegdata['class']
    X_train = X_train[testedSecond::11] if doSeconds else X_train
    y_train = y_train[testedSecond::11] if doSeconds else y_train
    X_train = filter_data(X_train, electrodes, minAppprox, maxApprox, minDets, maxDets, wantApprox, stats=_stats)
    X_train, y_train = shuffle(X_train, y_train, random_state=shafSeed)

    doPCA = _pca
    try:
        if doPCA:
            print("PCA running")
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            pca = PCA(n_components=_comp, random_state=48)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

    except Exception as e:
        print("PCA error")
        print(e)

    doICA = _ica
    try:
        if doICA:
            from sklearn.decomposition import FastICA
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            ica = FastICA(n_components=_comp, random_state=48)
            X_train = ica.fit_transform(X_train)
            X_test = ica.transform(X_test)
    except Exception as e:
        print("ICA err")
        print(e)

    return X_train, y_train, X_test, y_test



def random_grid():
    # Number of trees in random forest
    n_estimators = [100, 500, 1000]
    max_features = ['auto', 'sqrt']
    min_samples_split = [2, 5, 10, 50]
    min_samples_leaf = [10, 20]
    random_grid = {'n_estimators': n_estimators,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    return random_grid


# evaluates the model on test data and prints statistics
def evaluate(model, test_features, test_labels, X_train, y_train):
    print("Evaluation started" + str(time.time()))
    predictions = model.predict(test_features)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_score = model.score(X_train, y_train)
    test_score = model.score(test_features, test_labels)
    ret = confusion_matrix(test_labels, predictions)

    print(pd.crosstab(test_labels, predictions, rownames=['True'], colnames=['Predicted'], margins=True))
    print(classification_report(test_labels, predictions))
    print("Precision score: {}".format(precision_score(test_labels, predictions)))

    return (roc_auc, str(ret), str(train_score), str(test_score))


def classify(second, wavelet, electrodes, minApprox, maxApprox, minDets, maxDets, wantApprox,
             classifier=None,
             doSeconds=False,
             test_results=[], train_results=[],
             _pca=False,
             _stats=False,
             _ica=False, _comp=10,
             slidingSeconds=False):
    print("Classification started " + str(time.time()))


    X_train, y_train, X_test, y_test = get_datasets_ready(second, electrodes, minApprox, maxApprox, minDets, maxDets,
                                                          wantApprox, wavelet, doSeconds=doSeconds,
                                                          _stats=_stats,
                                                          _pca=_pca,
                                                          _ica=_ica,
                                                          _comp=_comp)


    print("Starting to fit" + str(time.time()))
    evaluated = ()
    try:
        model = classifier.fit(X_train, y_train)
        print("Fitting done" + str(time.time()))

        if slidingSeconds: # create a sliding window graph
            test_segments = []
            for custom_second in range(0,11):
                dummy_x, dummy_y,\
                X_test, y_test = get_datasets_ready(custom_second, electrodes,
                                                    minApprox, maxApprox, minDets,
                                                    maxDets, wantApprox, wavelet,
                                                    doSeconds=True)

                evaluated = evaluate(model, X_test, y_test, X_train, y_train)
                test_segments.append(evaluated[0]) # AUC
            plt.plot(test_segments)
            plt.xlabel('seconds')
            plt.show()

            sys.exit()

        evaluated = evaluate(model, X_test, y_test, X_train, y_train)
    except Exception as e:
        print(e)
    finally:

        return evaluated
        # print(evaluated)

    # shorthand for METAVO server
    """false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, evaluated[0])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

    train_pred = model.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)"""


def classifiers():
    svc = SVC(kernel='rbf', gamma='scale', random_state=48)
    nusvc = NuSVC(gamma=0.8, random_state=48)
    gbc = GradientBoostingClassifier(n_estimators=100,
                                     subsample=0.2,
                                     learning_rate=0.01,
                                     max_depth=5,
                                     verbose=True,
                                     random_state=48)
    rf = RandomForestClassifier(n_estimators=400,
                                criterion='gini',
                                verbose=True,
                                n_jobs=-1,
                                random_state=48,
                                max_features='auto',
                                )
    return [gbc, rf, svc, nusvc]


def single(log_file, filePrepare, classifier, testPair, results_file):
    print("starting single")
    discrete_mother_wavelets = pywt.wavelist(kind='discrete')  # all of them
    _pair = testPair
    _wavelet = 'bior2.4'
    _lesApr = 3
    _upApr = 5
    _lessDet = 2
    _upDet = 5
    _statistics = True
    _pds = False
    _fourier = False
    _approx = False
    _sample = 250
    _op = 250
    _electro = [True, False, True]
    _classifier = classifier
    startDrop = 10
    endDrop = 0


    current = add_together([_wavelet,
                            _lesApr,
                            _upApr,
                            _lessDet,
                            _upDet,
                            _statistics,
                            _fourier,
                            _approx,
                            _sample,
                            _op,
                            _electro,
                            _classifier,
                            _pair,
                            startDrop,
                            endDrop])
    print(current)

    result = classify(startDrop, _wavelet, _electro, _lesApr, _upApr, _lessDet, _upDet, _approx, classifier,
                      doSeconds=True)
    formattedResult = add_together(result)
    print(formattedResult)

    sys.exit()


def commendTesting():
    print("Testing initialized.")
    results_file = open("results_VICTOR__" + str(time.time()) + ".txt", "a")
    log_file = open("log_VICTOR_" + str(time.time()) + ".txt", "a")

    discrete_mother_wavelets = ['bior2.6',
                                'bior3.5',
                                'coif8',
                                'db21',
                                'db37',
                                'rbio3.5',
                                'rbio1.5',
                                'coif3',
                                'db11',
                                'db20',
                                'db28',
                                'sym9',
                                'sym2',
                                'db2']

    # discrete_mother_wavelets = pywt.wavelist(kind='discrete')  # all of them
    filePrepare = False
    pairs1 = [
        ["datasets/MartinTRA.csv", "datasets/MartinTES.csv"],
        ["datasets/sound_JJ1.csv", "datasets/sound_JJ2.csv"],
        ["datasets/sound_JJ1.csv", "datasets/sound_JJ3.csv"],
        ["datasets/sound_JJ2.csv", "datasets/sound_JJ3.csv"],
        ["datasets/visual_dominika_1704_open_imagine.csv", "datasets/visual_dominika_1704_open_s_gify.csv"],
        ["datasets/visual_dominika_1704_open_s_gify.csv", "datasets/visual_dominika_1704_open_imagine.csv"]
    ]

    pairs = [
        ["datasets/MartinTRA.csv", "datasets/MartinTES.csv"],
        ["dd/DominikaB_train.csv", "dd/DominikaB_test.csv"],
        ["dd/Dominika_train.csv", "dd/Dominika_test.csv"],
        ["dd/Charlie_train.csv", "dd/Charlie_test.csv"],
        ["dd/Bravo_train.csv", "dd/Bravo_test.csv"],
        ["dd/Foxtrot_train.csv", "dd/Foxtrot_test.csv"],
        ["dd/Echo_train.csv", "dd/Echo_test.csv"],
        ["dd/Jan_train.csv", "dd/Jan_test.csv"],
        ["dd/Martin_train.csv", "dd/Martin_test.csv"],
        ["dd/Lima_train.csv", "dd/Lima_test.csv"],
        ["dd/Oscar_train.csv", "dd/Oscar_test.csv"],
        ["dd/Victor_train.csv", "dd/Victor_test.csv"],
        ["dd/Zulu_train.csv", "dd/Zulu_test.csv"],
    ]

    martites1 = [
        ["datasets/MartinTRA.csv", "datasets/MartinTES.csv"],

    ]

    martites = [
        ["renamed/serenity/Martin_train.csv", "renamed/serenity/Martin_test.csv"],
    ]

    estimators = [100, 250, 400,500, 750, 1000, 2000]
    lesserApprox = [3]
    upperApprox = [3]
    lesserDetails = [3]
    upperDetails = [3]
    statistics = [True]
    plot_pds = [False]
    _fft = [False]
    wantApprox = [False]
    _samples = [250]
    _overlap = [250]
    startDrops = [4]
    endDrops = [0]
    electrodes = [
        [True, False, True],  # C3 C4
    ]
    classifiers_arr = classifiers()
    pcas = [True]
    icas = [True]
    ica = True

    classifiers_arr = [classifiers_arr[0]]
    # single(log_file, False, classifiers_arr[0], martites[0], results_file)

    discrete_mother_wavelets = ['bior3.1_classic']  ######################################
    trainresults = []
    test_results = []

    xvals = []
    for _wavelet in discrete_mother_wavelets:
        for _lesApr in lesserApprox:
            for _upApr in upperApprox:
                for _lessDet in lesserDetails:
                    for _upDet in upperDetails:
                        for _approx in wantApprox:
                            for _electro in electrodes:
                                for stats in statistics:
                                    for _startDrop in startDrops:
                                        for pca in pcas:
                                            for snus in icas:
                                                current = add_together([_wavelet,
                                                                        _lesApr,
                                                                        _upApr,
                                                                        _lessDet,
                                                                        _upDet,
                                                                        _approx,
                                                                        _electro,
                                                                        _startDrop,
                                                                        stats,
                                                                        pca,
                                                                        ica
                                                                        ])
                                                print(current)

                                                try:
                                                    # single/individual manual classification
                                                    _wavelet = 'bior3.1_classic'
                                                    _lessDet = 3
                                                    _upDet = 4
                                                    _approx=False
                                                    _lesApr = 3
                                                    _upApr = 4
                                                    model = classifiers_arr[0] ; print(model)
                                                    result = classify(_startDrop, _wavelet, _electro, _lesApr,
                                                                      _upApr, _lessDet, _upDet, _approx, model,
                                                                      doSeconds=False,
                                                                      _stats=False,
                                                                      _pca=False,
                                                                      _ica=False,
                                                                      _comp=6)

                                                    current = add_together([_wavelet,
                                                                            _lesApr,
                                                                            _upApr,
                                                                            _lessDet,
                                                                            _upDet,
                                                                            _approx,
                                                                            _electro,
                                                                            _startDrop,
                                                                            stats,
                                                                            pca,
                                                                            ica,
                                                                            snus,
                                                                            snus,
                                                                            snus
                                                                            ])
                                                    print(current)
                                                    print(result)

                                                    test_results.append(result[0])
                                                    trainresults.append(result[2])
                                                    xvals.append(_wavelet)

                                                    formattedResult = add_together(result)
                                                    write_record(results_file, "Result;;" + str(
                                                        formattedResult + current).strip().replace(
                                                        "\n",
                                                        ","))
                                                    write_record(log_file,
                                                                 "RESULT;;" + str(
                                                                     result).strip().replace(
                                                                     "\n", ","))
                                                    print((formattedResult + current).replace("\n", "").strip())
                                                    print(
                                                        "===================================================================")
                                                    write_record(log_file,
                                                                 "--------------------------------------------------------------------------------------------------------------------------------------")
                                                    current = ""
                                                except Exception as e:
                                                    write_record(log_file, str(e))
                                                    print(e)
                                                    exc_type, exc_obj, exc_tb = sys.exc_info()

                                                    fname = os.path.split(
                                                        exc_tb.tb_frame.f_code.co_filename)[1]
                                                    print(exc_type, fname, exc_tb.tb_lineno)
                                                    print(traceback.format_exc())
                                                    pass

    results_file.close()
    log_file.close()

    # pt.myplt(xvals, trainresults, test_results) # for plotting results into a graph


def add_together(arr):
    s = ""
    for item in arr:
        s += str(item).strip() + ";"
    return s


def write_record(f, rec):
    # print("NOT WRITING LOGS!")
    f.write(";" + str(rec).strip() + "; time:" + str(time.time()) + "\n")
    f.flush()
    pass


if __name__ == '__main__':
    commendTesting()
    # whitening()


def test_best(X_train, y_train, X_test, y_test, filehandle=None):
    grid = random_grid()
    for est in grid['n_estimators']:
        for feat in grid['max_features']:
            for depth in grid['max_depth']:
                for mins in grid['min_samples_split']:
                    for minl in grid['min_samples_leaf']:
                        for boots in grid['bootstrap']:
                            clss = RandomForestClassifier(n_estimators=est,
                                                          criterion='entropy',
                                                          verbose=True,
                                                          n_jobs=-1,
                                                          random_state=48,
                                                          max_features=feat,
                                                          max_depth=depth,
                                                          min_samples_split=mins,
                                                          min_samples_leaf=minl,
                                                          bootstrap=boots)
                            model = clss.fit(X_train, y_train)
                            acc = evaluate(model, X_test, y_test, X_train, y_train)

                            sss = str(str(acc) + ";;" + clss)
                            filehandle.write(sss + "\n")
                            filehandle.flush()
                            print(sss)
