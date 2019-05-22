import pprint
import sys
import traceback
import errno
import os
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simps
from scipy.signal import welch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import os

xgbon = False
if xgbon:
    import xgboost as xgb

sys.path.insert(0, 'software/python-3.4.1/gcc/lib/python3.4/site-packages/')  # pro doom server
import matplotlib as mpl
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
from sklearn.utils import shuffle
import csv
import datetime
from collections import Counter
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, roc_curve
import time
from sklearn import preprocessing
import scipy.io
from sklearn.metrics import roc_curve, auc

from ModulesEEG import EEGAnalysis
import PlotTools

final_data = []
final_labels = []


def prepareFile(fn="clean_dataset\\1556726380_dataset_Martin.csv", test=False, wvlt='sym9', outName='train.csv',
                lesserApprox=3, upperApprox=6, lesserDetails=3, upperDetails=6,
                doStatistics=True, plot_psd=False,
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
        eeg = EEGAnalysis(fileData)
        eeg.extractCsvChannelIntoArray('c3')  # no matter the channel, the array is done
        eeg.transformInputData(_samples, False, startDrop, endDrop)  # dont drop any data
        print("asd5")

        C3_ON = electrodes[0]
        CZ_ON = electrodes[1]
        C4_ON = electrodes[2]
        
        print("Lets go ona and compute the classes")
        eeg.compute_class(0, eeg.data_left, wvlt, lesserApprox, upperApprox, lesserDetails, upperDetails,
                          doStatistics, plot_psd, wholeArrC3, wholeArrC4, wholeArrCz,
                          c3_d, c4_d, cz_d, _fft, c3_feat,
                          c4_feat, cz_feat, wantApprox, _samples, _overlap, [C3_ON, CZ_ON, C4_ON])
        eeg.compute_class(1, eeg.data_right, wvlt, lesserApprox, upperApprox, lesserDetails, upperDetails,
                          doStatistics, plot_psd, wholeArrC3, wholeArrC4, wholeArrCz,
                          c3_d, c4_d, cz_d, _fft, c3_feat,
                          c4_feat, cz_feat, wantApprox, _samples, _overlap, [C3_ON, CZ_ON, C4_ON])


        print("saving from " + fileData)
        print("saving to " + outName)
        with open(outName, "w", newline='') as f:
            fieldnames = ['index', 'class']
            dataset_writer = csv.writer(f, delimiter=',')

            columnsWidth = 0
            rowsHeight = 0
            if doStatistics:
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
                    avgIndex += len(c3_d[i]) if doStatistics else len(c3_feat[i])
                if C4_ON and not C3_ON:
                    avgIndex += len(c4_d[i]) if doStatistics else len(c4_feat[i])
                if C3_ON and C4_ON:
                    avgIndex += len(c4_d[i]) if doStatistics else len(c4_feat[i])

            _avg = int(avgIndex / rowsHeight)

            while i < rowsHeight:
                label = eeg.finalLabels[i]
                if doStatistics:
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

                    # print("array len c3 " + str(len(c3_feat[i])))
                    if C3_ON:
                        arry.extend(c3_feat[i])
                    # print("array len c4 " + str(len(c4_feat[i])))
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



def filter_data(arrin, electrodes, minAprox, maxAprox, minDet, maxDet, wantAprox):
    C3_ON = electrodes[0]
    CZ_ON = electrodes[1]
    C4_ON = electrodes[2]

    newsec = []
    for i in range(len(arrin)):
        second = arrin.iloc[[i]].values[0]
        arr = []
        if C3_ON:
            if wantAprox: arr.extend(get_filered(second, minAprox, maxAprox, 0))
            arr.extend(get_filered(second, minDet, maxDet, 1532))
        if C4_ON:
            if wantAprox: arr.extend(get_filered(second, minAprox, maxAprox, 3064))
            arr.extend(get_filered(second, minDet, maxDet, 4596))
        if CZ_ON:
            if wantAprox: arr.extend(get_filered(second, minAprox, maxAprox, 6128))
            arr.extend(get_filered(second, minDet, maxDet, 7660))
        newsec.append(arr)
    return newsec


def get_filered(second, _min, _max, electrodeStart):
    ixt = [0, 250, 500, 750, 1000, 1266]
    ixs = [250, 500, 750, 1000, 1266, 1532]
    res = []

    for i, val in enumerate(ixt):
        if _min <= i <= _max:
            res.extend(second[ixt[i] + electrodeStart:ixs[i] + electrodeStart])
    return res


def get_datasets_ready(testedSecond=1):
    print("Getting dataset ready" + str(time.time()))
    shafSeed = 48
    eegdata = pd.read_csv("train.csv")
    # eegdata = eegdata.dropna()  # drop useless values
    eegdata = eegdata.fillna(0)

    eegtest = pd.read_csv("test.csv")
    # eegtest = eegtest.dropna()
    eegtest = eegtest.fillna(0)
    # test
    X_test = eegtest.drop('class', axis=1).drop('index', axis=1)
    y_test = eegtest['class']

    X_test = X_test[testedSecond::11]
    y_test = y_test[testedSecond::11]
    X_test = filter_data(X_test, [1,0,1], 0,0,3,5,0)
    X_test, y_test = shuffle(X_test, y_test, random_state=shafSeed)

    # train
    X_train = eegdata.drop('class', axis=1).drop('index', axis=1)
    y_train = eegdata['class']
    X_train = X_train[testedSecond::11]
    y_train = y_train[testedSecond::11]
    X_train = filter_data(X_train, [1, 0, 1], 0, 0, 3, 5, 0)
    X_train, y_train = shuffle(X_train, y_train, random_state=shafSeed)

    doPCA = False

    try:
        if doPCA:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            pca = PCA(n_components=1)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

    except Exception as e:
        print("PCA error")
        print(e)

    return X_train, y_train, X_test, y_test


def random_grid():
    n_estimators = [int(x) for x in np.linspace(start=300, stop=900, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    return random_grid


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


def classify(classifier=SVC(kernel='rbf', gamma='scale'), test_results=[], train_results=[], paramsDict={}):
    print("Classification started " + str(time.time()))

    for i in range(0, 11):

        print("second: " + str(i))
        X_train, y_train, X_test, y_test = get_datasets_ready(i)

        print("Starting to fit" + str(time.time()))
        evaluated = ()
        try:
            model = classifier.fit(X_train, y_train)
            print("Fitting done" + str(time.time()))
            evaluated = evaluate(model, X_test, y_test, X_train, y_train)
        except Exception as e:
            print(e)
        finally:
            # return evaluated
            print(evaluated)


def classifiers():
    svc = SVC(kernel='rbf', gamma=0.6, probability=False, random_state=48)
    svc = SVC(random_state=48)
    nusvc = NuSVC(gamma=0.5)
    gbc = GradientBoostingClassifier(n_estimators=1200, learning_rate=0.05, verbose=True)
    ada = AdaBoostClassifier(n_estimators=700, learning_rate=0.05)
    mpl = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(250, 4), random_state=1)
    # seq = seq_nn()

    rf = RandomForestClassifier(n_estimators=390,
                                criterion='entropy',
                                verbose=True,
                                n_jobs=-1,
                                random_state=48,
                                max_features='auto',
                                max_depth=20,
                                min_samples_split=100,
                                min_samples_leaf=15,
                                bootstrap=True)

    rf = RandomForestClassifier(n_estimators=390,
                                criterion="entropy",
                                max_depth=10,
                                min_samples_split=200,
                                min_samples_leaf=20,
                                min_weight_fraction_leaf=0.0,
                                max_features="log2",
                                max_leaf_nodes=None,
                                min_impurity_decrease=0,
                                min_impurity_split=None,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=-1,
                                random_state=48,
                                verbose=True,
                                warm_start=False)

    from sklearn.svm import LinearSVC
    return [gbc, rf, gbc, ada, svc, nusvc, mpl]


def single(log_file, filePrepare, classifier, testPair, results_file):
    print("starting single")
    discrete_mother_wavelets = pywt.wavelist(kind='discrete')  # all of them
    _pair = testPair
    _wavelet = 'bior3.1'
    _lesApr = 0
    _upApr = 6
    _lessDet = 0
    _upDet = 6
    _doStatistics = True # FAlSE gave the best results so far
    _pds = False
    _fourier = False
    _approx = True
    _sample = 250
    _op = 250
    _electro = [True, False, True]
    _classifier = classifier
    startDrop = 0
    endDrop = 0
    print("loading")
    print(testPair)
    discrete_mother_wavelets = [_wavelet]
    try:
        if filePrepare:
            print("Lets prepare some files")
            for _wavelet in discrete_mother_wavelets:
                prepareFile(fn=_pair[0], wvlt=_wavelet,
                            outName=_wavelet+"/"+'train.csv',
                            lesserApprox=_lesApr,
                            upperApprox=_upApr,
                            lesserDetails=_lessDet,
                            upperDetails=_upDet,
                            doStatistics=_doStatistics,
                            plot_psd=_pds,
                            _fft=_fourier,
                            wantApprox=_approx,
                            _samples=_sample,
                            _overlap=_op,
                            electrodes=_electro,
                            startDrop=startDrop,
                            endDrop=endDrop)
                prepareFile(fn=_pair[1], wvlt=_wavelet,
                            outName=_wavelet+"/"+'test.csv',
                            lesserApprox=_lesApr,
                            upperApprox=_upApr,
                            lesserDetails=_lessDet,
                            upperDetails=_upDet,
                            doStatistics=_doStatistics,
                            plot_psd=_pds,
                            _fft=_fourier,
                            wantApprox=_approx,
                            _samples=_sample,
                            _overlap=_op,
                            electrodes=_electro,
                            startDrop=startDrop,
                            endDrop=endDrop
                            )
            print("Files generated: " + str(time.time()))
            sys.exit(0)
        else:
            print(
                "NOT PREPARING FILE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        current = add_together([_wavelet,
                                _lesApr,
                                _upApr,
                                _lessDet,
                                _upDet,
                                _doStatistics,
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
        result = classify(_classifier)
        formattedResult = add_together(result)
        write_record(results_file, formattedResult + current)
        write_record(log_file, "RESULT: " + str(result))
        print(formattedResult + current)
        print("===================================================================")
        write_record(log_file,
                     "--------------------------------------------------------------------------------------------------------------------------------------")

    except Exception as e:
        write_record(log_file, str(e))

    sys.exit()


def commendTesting():
    print("Testing initialized.")
    results_file = open("results_TEST_rbio1.5_approxne__" + str(time.time()) + ".txt", "a")
    log_file = open("log_TEST_rbio1.5_approxne_" + str(time.time()) + ".txt", "a")

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
    discrete_mother_wavelets = pywt.wavelist(kind='discrete')  # all of them
    discrete_mother_wavelets = ['bior3.5']
    filePrepare = True
    pairs = [
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

    martites = [
        ["datasets/MartinTRA.csv", "datasets/MartinTES.csv"],

    ]

    naam = "Oscar"
    try:
        naam = sys.argv[1]
    except Exception as e:
        print(e)
        print("Enter a value like: Oscar, Echo, Lima, etc.")
        sys.exit()
    martites = [
        ["../Datasety/"+naam+"_train.csv",
         "../Datasety/"+naam+"_test.csv"],
    ]

    estimators = [100, 500, 1000, 2000]
    lesserApprox = [1]
    upperApprox = [4]
    # lesserDetails = np.arange(1,7,1)
    # upperDetails=np.arange(1,8,1)
    lesserDetails = [3]
    upperDetails = [6]
    doStatist = [True, False]
    plot_pds = [False]
    _fft = [False]
    wantApprox = [False]
    _samples = [250]
    _overlap = [250]
    startDrops = [0]
    endDrops = [0]
    electrodes = [
        [True, False, True],  # C3 C4
        # [True, False, False], # C3
        # [False, False, True] # C4
    ]
    classifiers_arr = classifiers()

    classifiers_arr = [classifiers_arr[0]]
    single(log_file, True, classifiers_arr[0], martites[0], results_file)


    for _wavelet in discrete_mother_wavelets:
        for _lesApr in lesserApprox:
            for _upApr in upperApprox:
                for _lessDet in lesserDetails:
                    for _upDet in upperDetails:
                        for _wildo in doStatist:
                            for _pds in plot_pds:
                                for _fourier in _fft:
                                    for _approx in wantApprox:
                                        for _sample in _samples:
                                            for _op in _overlap:
                                                for _electro in electrodes:
                                                    for _startDrop in startDrops:
                                                        for _endDrop in endDrops:
                                                            for _pair in pairs:
                                                                for _classifier in classifiers_arr:
                                                                    current = add_together([_wavelet,
                                                                                            _lesApr,
                                                                                            _upApr,
                                                                                            _lessDet,
                                                                                            _upDet,
                                                                                            _wildo,
                                                                                            _fourier,
                                                                                            _approx,
                                                                                            _sample,
                                                                                            _op,
                                                                                            _electro,
                                                                                            _classifier,
                                                                                            _pair,
                                                                                            _startDrop,
                                                                                            _endDrop])
                                                                    print(current)

                                                                    try:
                                                                        if filePrepare:
                                                                            prepareFile(fn=_pair[0], wvlt=_wavelet,
                                                                                        outName='train.csv',
                                                                                        lesserApprox=_lesApr,
                                                                                        upperApprox=_upApr,
                                                                                        lesserDetails=_lessDet,
                                                                                        upperDetails=_upDet,
                                                                                        doStatistics=_wildo,
                                                                                        plot_psd=_pds,
                                                                                        _fft=_fourier,
                                                                                        wantApprox=_approx,
                                                                                        _samples=_sample,
                                                                                        _overlap=_op,
                                                                                        electrodes=_electro,
                                                                                        startDrop=_startDrop,
                                                                                        endDrop=_endDrop)
                                                                            prepareFile(fn=_pair[1], wvlt=_wavelet,
                                                                                        outName='test.csv',
                                                                                        lesserApprox=_lesApr,
                                                                                        upperApprox=_upApr,
                                                                                        lesserDetails=_lessDet,
                                                                                        upperDetails=_upDet,
                                                                                        doStatistics=_wildo,
                                                                                        plot_psd=_pds,
                                                                                        _fft=_fourier,
                                                                                        wantApprox=_approx,
                                                                                        _samples=_sample,
                                                                                        _overlap=_op,
                                                                                        electrodes=_electro,
                                                                                        startDrop=_startDrop,
                                                                                        endDrop=_endDrop)
                                                                        else:
                                                                            print(
                                                                                "NOT PREPARING FILE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                                                                        result = classify(_classifier)
                                                                        formattedResult = add_together(result)
                                                                        write_record(results_file, "Result:: " + str(
                                                                            formattedResult + current).strip().replace(
                                                                            "\n",
                                                                            ","))
                                                                        write_record(log_file,
                                                                                     "RESULT: " + str(
                                                                                         result).strip().replace(
                                                                                         "\n", ","))
                                                                        print(formattedResult + current)
                                                                        print(
                                                                            "===================================================================")
                                                                        write_record(log_file,
                                                                                     "--------------------------------------------------------------------------------------------------------------------------------------")

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
    trainresults = []
    test_results = []
    xvals = []

    # myplt(xvals, trainresults, test_results)


def add_together(arr):
    s = ""
    for item in arr:
        s += str(item).strip() + ";"
    return s


def write_record(f, rec):
    f.write(";" + str(rec).strip() + "; time:" + str(time.time()) + "\n")
    f.flush()


if __name__ == '__main__':
    commendTesting()
    # whitening()


def test_best(X_train, y_train, X_test, y_test, filehandle=None):
    grid = random_grid()
    # 0.674864243943;100;auto;30;100;20;False;[[98 46] [44 89]];0.858347386172;0.675090252708;
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
