
import numpy as np
from matplotlib import pyplot as plt


class DataStorage:

    def __init__(self, run=False):
        """initializing data storage"""
        self.__name_mlp = 'Maximum Likelihood Predictor'
        self.__name_nbc = 'Naive Bayes Classifier'
        self.__name_lrc = 'Logistic Regression Classifier'
        self.__name_dtc = 'Decision Tree Classifier'
        self.__name_rfc = 'Random Forests Classifier'
        self.__name_svc = 'Support Vector Machines Classifier'
        self.__name_nnc = 'Nearest Neighbors Classifier'
        self.__name_kmm = 'Means Clustering Classifier'
        self.__name_gmm = 'Gaussian Mixture Models Classifier'
        self.__show = run
        return

    def get_data_mlp(self) -> tuple:
        """maximum likelihood predictor error/runtime"""
        err_0 = [0.38939999999999997,
                 0.24012,
                 0.09266700000000005,
                 0.025567000000000006,
                 0.0021090000000000275,
                 9.999999999998899e-05,
                 1.2000000000012001e-05,
                 8.000000000008e-06,
                 0.0,
                 0.0]
        err_1 = [0.37922,
                 0.232665,
                 0.10059300000000004,
                 0.026212000000000013,
                 0.0026080000000000547,
                 0.00032399999999999096,
                 4.8000000000048004e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.38424800000000003,
                 0.23358900000000005,
                 0.09775699999999998,
                 0.022916999999999965,
                 0.0022600000000000398,
                 0.00015900000000002024,
                 1.799999999996249e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_3 = [0.38414800000000004,
                 0.23629900000000004,
                 0.102209,
                 0.024878999999999984,
                 0.0022689999999999655,
                 0.00023700000000004273,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.382772,
                 0.23834200000000005,
                 0.10274399999999995,
                 0.021561000000000052,
                 0.0027530000000000054,
                 0.00014499999999995072,
                 9.000000000036756e-06,
                 2.9999999999752447e-06,
                 0.0,
                 1.0000000000287557e-06]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [05.52923703,
                  05.16168499,
                  05.11924005,
                  05.61225891,
                  06.18116212,
                  05.12005305]
        time_1 = [05.62633109,
                  05.02867818,
                  04.99123001,
                  05.01003075,
                  05.00932717,
                  05.02505112]
        time_2 = [05.28055882,
                  05.05696106,
                  04.98890781,
                  05.30412817,
                  05.40379596,
                  04.97286820]
        time_3 = [05.81717134,
                  04.97628117,
                  04.99082804,
                  04.98211694,
                  04.98160529,
                  04.99857616]
        time_4 = [05.42290592,
                  05.40907192,
                  05.09239912,
                  04.97821093,
                  04.97702503,
                  04.95435309]
        time = np.array(np.concatenate((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_mlp, error, time

    def get_data_nbc(self) -> tuple:
        """naive Bayes classifier error/runtime"""
        err_0 = [0.38957299999999995,
                 0.24039200000000005,
                 0.093024,
                 0.02588199999999996,
                 0.0022339999999999582,
                 0.0001100000000000545,
                 1.399999999995849e-05,
                 9.99999999995449e-06,
                 1.0000000000287557e-06,
                 0.0]
        err_1 = [0.37921499999999997,
                 0.23268999999999995,
                 0.10066200000000003,
                 0.02624599999999999,
                 0.0026199999999999557,
                 0.00033199999999999896,
                 4.599999999999049e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.384493,
                 0.23395900000000003,
                 0.09828300000000001,
                 0.023317000000000032,
                 0.0023940000000000072,
                 0.00019899999999994922,
                 2.4000000000024002e-05,
                 4.000000000004e-06,
                 0.0,
                 0.0]
        err_3 = [0.38417,
                 0.23632299999999995,
                 0.10226299999999999,
                 0.024927000000000032,
                 0.0022699999999999942,
                 0.00024299999999999322,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.382787,
                 0.23834200000000005,
                 0.10274899999999998,
                 0.02157500000000001,
                 0.00275000000000003,
                 0.00014499999999995072,
                 9.000000000036756e-06,
                 1.999999999946489e-06,
                 0.0,
                 1.0000000000287557e-06]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [06.38303709,
                  06.09069109,
                  05.66709995,
                  05.67149401,
                  07.07361889,
                  12.49591875]
        time_1 = [5.94973588,
                  5.60216379,
                  5.62797809,
                  5.72952604,
                  6.21111107,
                  11.94398808]
        time_2 = [05.90092063,
                  05.62883306,
                  05.63064098,
                  05.70871592,
                  06.23375797,
                  12.22252011]
        time_3 = [05.96813297,
                  05.63997817,
                  05.64475203,
                  05.71107125,
                  07.10176110,
                  14.23913312]
        time_4 = [06.37981796,
                  05.62256098,
                  05.90896511,
                  05.98498297,
                  06.52529812,
                  13.04716086]
        time = np.array(np.mean((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_nbc, error, time

    def get_data_lrc(self) -> tuple:
        """logistic regression classifier error/runtime"""
        err_0 = [0.389532,
                 0.24030700000000005,
                 0.09289499999999995,
                 0.025774999999999992,
                 0.0021769999999999845,
                 0.00010500000000002174,
                 1.2000000000012001e-05,
                 9.99999999995449e-06,
                 1.0000000000287557e-06,
                 0.0]
        err_1 = [0.37930299999999995,
                 0.23274300000000003,
                 0.10071600000000003,
                 0.026336000000000026,
                 0.0026399999999999757,
                 0.0003389999999999782,
                 4.8000000000048004e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.38442600000000005,
                 0.23384700000000003,
                 0.09811099999999995,
                 0.023156999999999983,
                 0.002346999999999988,
                 0.00017999999999995797,
                 2.2999999999995246e-05,
                 2.9999999999752447e-06,
                 0.0,
                 0.0]
        err_3 = [0.38416300000000003,
                 0.236309,
                 0.10222799999999999,
                 0.024907000000000012,
                 0.002263000000000015,
                 0.00023899999999998922,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.38279399999999997,
                 0.238359,
                 0.10275699999999999,
                 0.021584999999999965,
                 0.0027559999999999807,
                 0.000144000000000033,
                 9.000000000036756e-06,
                 1.999999999946489e-06,
                 0.0,
                 1.0000000000287557e-06]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [05.86358404,
                  05.18709111,
                  05.40122390,
                  06.70335484,
                  06.99454713,
                  30.30031896]
        time_1 = [06.01590610,
                  05.40210915,
                  06.37035584,
                  06.40896606,
                  08.53482294,
                  29.90523195]
        time_2 = [05.60408211,
                  05.71548510,
                  05.98794913,
                  07.24933410,
                  07.41941285,
                  30.31748819]
        time_3 = [05.17387605,
                  05.47211695,
                  05.63621712,
                  06.83456802,
                  07.65061188,
                  28.94552112]
        time_4 = [05.58673501,
                  05.33242130,
                  06.59896994,
                  05.58651781,
                  07.33440590,
                  27.42983007]
        time = np.array(np.mean((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_lrc, error, time

    def get_data_dtc(self) -> tuple:
        """decision tree classifier error/runtime"""
        err_0 = [0.38957200000000003,
                 0.24038000000000004,
                 0.09301300000000001,
                 0.025850999999999957,
                 0.0022320000000000118,
                 0.0001100000000000545,
                 1.3000000000040757e-05,
                 1.0999999999983245e-05,
                 1.0000000000287557e-06,
                 0.0]
        err_1 = [0.37924500000000005,
                 0.23266200000000004,
                 0.10060899999999995,
                 0.02622000000000002,
                 0.002610000000000001,
                 0.0003229999999999622,
                 4.8000000000048004e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.384973,
                 0.23463,
                 0.09912299999999996,
                 0.02394099999999999,
                 0.0025960000000000427,
                 0.0002689999999999637,
                 3.399999999997849e-05,
                 6.999999999979245e-06,
                 1.0000000000287557e-06,
                 0.0]
        err_3 = [0.384165,
                 0.23630099999999998,
                 0.10228099999999996,
                 0.024922,
                 0.002267000000000019,
                 0.00024100000000004673,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.383053,
                 0.23874099999999998,
                 0.10338999999999998,
                 0.022040999999999977,
                 0.002946000000000004,
                 0.00017900000000004024,
                 1.7000000000044757e-05,
                 4.000000000004e-06,
                 1.999999999946489e-06,
                 0.0]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [06.58602118,
                  06.86498094,
                  06.26244688,
                  06.62626314,
                  08.56289005,
                  43.76612091]
        time_1 = [06.24585104,
                  06.22860312,
                  06.26018286,
                  06.97714472,
                  09.18052411,
                  44.44010210]
        time_2 = [06.72292924,
                  06.20027494,
                  06.97790813,
                  07.17930984,
                  09.26475120,
                  43.14399791]
        time_3 = [06.11175013,
                  06.12165236,
                  05.84791708,
                  06.54275203,
                  09.22229028,
                  48.37857199]
        time_4 = [06.23834896,
                  05.99268699,
                  06.33106709,
                  06.42494893,
                  08.59175801,
                  41.43895602]
        time = np.array(np.mean((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_dtc, error, time

    def get_data_rfc(self) -> tuple:
        """random forests classifier error/runtime"""
        err_0 = [0.38957299999999995,
                 0.24036500000000005,
                 0.09300799999999998,
                 0.02585499999999996,
                 0.0022290000000000365,
                 0.00011099999999997223,
                 1.3000000000040757e-05,
                 1.0999999999983245e-05,
                 1.0000000000287557e-06,
                 0.0]
        err_1 = [0.379251,
                 0.23265599999999997,
                 0.100611,
                 0.026221999999999968,
                 0.00261100000000003,
                 0.0003229999999999622,
                 4.8000000000048004e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.384941,
                 0.23459700000000006,
                 0.09907100000000002,
                 0.023930000000000007,
                 0.0025889999999999525,
                 0.000268000000000046,
                 3.2999999999949736e-05,
                 6.999999999979245e-06,
                 1.0000000000287557e-06,
                 0.0]
        err_3 = [0.38417599999999996,
                 0.23631100000000005,
                 0.10226400000000002,
                 0.02491500000000002,
                 0.0022680000000000478,
                 0.00023799999999996047,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.382914,
                 0.23857399999999995,
                 0.10311899999999996,
                 0.02186100000000002,
                 0.002882000000000051,
                 0.00016599999999999948,
                 1.3000000000040757e-05,
                 1.999999999946489e-06,
                 0.0,
                 0.0]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [9.772062060,
                  10.58162212,
                  10.23710799,
                  11.68104887,
                  18.05361795,
                  136.0953152]
        time_1 = [11.14723992,
                  11.76904917,
                  12.03093290,
                  13.16373992,
                  19.19387698,
                  146.4961901]
        time_2 = [11.48275089,
                  10.91608906,
                  10.75092721,
                  11.63044190,
                  17.17749023,
                  129.331210]
        time_3 = [9.786290170,
                  9.627894880,
                  10.14560103,
                  11.39658308,
                  16.63728380,
                  126.5340910]
        time_4 = [9.189521310,
                  9.428395990,
                  11.08337379,
                  11.46144319,
                  17.07096887,
                  123.9435680]
        time = np.array(np.mean((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_rfc, error, time

    def get_data_svc(self) -> tuple:
        """support vector machines classifier error/runtime"""
        err_0 = [0.389582,
                 0.24047399999999997,
                 0.09308099999999997,
                 0.025939999999999963,
                 0.002255000000000007,
                 0.00011399999999994748,
                 1.399999999995849e-05,
                 1.3000000000040757e-05,
                 1.0000000000287557e-06,
                 0.0]
        err_1 = [0.379219,
                 0.23272800000000005,
                 0.10065900000000005,
                 0.026251000000000024,
                 0.0026119999999999477,
                 0.0003260000000000485,
                 4.8000000000048004e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.384856,
                 0.234553,
                 0.09892699999999999,
                 0.023797999999999986,
                 0.002565999999999957,
                 0.00024500000000005073,
                 2.999999999997449e-05,
                 5.999999999950489e-06,
                 1.0000000000287557e-06,
                 1.0000000000287557e-06]
        err_3 = [0.384197,
                 0.23637200000000003,
                 0.10230700000000004,
                 0.02492300000000003,
                 0.0022809999999999775,
                 0.00023700000000004273,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.38276699999999997,
                 0.238417,
                 0.10278200000000004,
                 0.021584999999999965,
                 0.0027610000000000134,
                 0.00014599999999997948,
                 9.000000000036756e-06,
                 1.999999999946489e-06,
                 0.0,
                 1.0000000000287557e-06]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [10.16640997,
                  10.97731519,
                  12.67191982,
                  13.22019196,
                  19.37110877,
                  143.2974789]
        time_1 = [11.44261098,
                  12.11190724,
                  12.65949488,
                  14.80235004,
                  22.05263305,
                  145.8489733]
        time_2 = [10.50752997,
                  11.73149514,
                  13.95262098,
                  14.47972727,
                  20.95897603,
                  155.9046261]
        time_3 = [10.52782798,
                  10.72704101,
                  13.85724282,
                  13.39441705,
                  22.35930967,
                  154.3177910]
        time_4 = [13.87285089,
                  11.24556494,
                  14.32515192,
                  14.96226573,
                  24.04077792,
                  150.5950589]
        time = np.array(np.mean((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_svc, error, time

    def get_data_nnc(self) -> tuple:
        """nearest neighbors classifier error/runtime"""
        err_0 = [0.389524,
                 0.24031999999999998,
                 0.09290500000000002,
                 0.025808000000000053,
                 0.0022170000000000245,
                 0.0001100000000000545,
                 1.3000000000040757e-05,
                 1.0999999999983245e-05,
                 0.0,
                 0.0]
        err_1 = [0.379263,
                 0.232765,
                 0.10070199999999996,
                 0.026329000000000047,
                 0.0026370000000000005,
                 0.0003389999999999782,
                 4.599999999999049e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.38446,
                 0.23390299999999997,
                 0.098271,
                 0.023252999999999968,
                 0.002371000000000012,
                 0.00019400000000002748,
                 2.2999999999995246e-05,
                 4.000000000004e-06,
                 0.0,
                 0.0]
        err_3 = [0.38415,
                 0.236383,
                 0.10231299999999999,
                 0.024939000000000044,
                 0.002275000000000027,
                 0.00024400000000002198,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.38281299999999996,
                 0.23841500000000004,
                 0.10279899999999997,
                 0.021592999999999973,
                 0.002762000000000042,
                 0.00014899999999995472,
                 9.000000000036756e-06,
                 1.999999999946489e-06,
                 0.0,
                 0.0]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [584.61974669,
                  510.09263897,
                  525.78326726,
                  551.93049502,
                  542.97495699,
                  717.42673111]
        time_1 = [547.22068024,
                  528.84433818,
                  550.59464598,
                  559.3739028,
                  542.25543594,
                  759.08158898]
        time_2 = [540.63957787,
                  540.31299424,
                  510.89212179,
                  532.47241092,
                  543.13045096,
                  670.62258792]
        time_3 = [547.25457692,
                  557.94879675,
                  584.55458999,
                  775.72203398,
                  557.18929720,
                  683.66764593]
        time_4 = [521.53018999,
                  521.1739521,
                  523.91868806,
                  527.47346091,
                  571.16548705,
                  698.92083979]
        time = np.array(np.mean((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_nnc, error, time

    def get_data_kmm(self) -> tuple:
        """means clustering classifier error/runtime"""
        err_0 = [0.38939999999999997,
                 0.24012,
                 0.09266700000000005,
                 0.025567000000000006,
                 0.0021090000000000275,
                 9.999999999998899e-05,
                 1.2000000000012001e-05,
                 8.000000000008e-06,
                 0.0,
                 0.0]
        err_1 = [0.37922,
                 0.232665,
                 0.10059300000000004,
                 0.026212000000000013,
                 0.0026080000000000547,
                 0.00032399999999999096,
                 4.8000000000048004e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.38424899999999995,
                 0.23358900000000005,
                 0.09775699999999998,
                 0.022916999999999965,
                 0.0022600000000000398,
                 0.00015900000000002024,
                 1.799999999996249e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_3 = [0.38414800000000004,
                 0.23629900000000004,
                 0.102209,
                 0.024878999999999984,
                 0.0022689999999999655,
                 0.00023700000000004273,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.382772,
                 0.23834299999999997,
                 0.10274399999999995,
                 0.021561000000000052,
                 0.0027530000000000054,
                 0.00014499999999995072,
                 9.000000000036756e-06,
                 2.9999999999752447e-06,
                 0.0,
                 1.0000000000287557e-06]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [43.93173695,
                  45.18520212,
                  41.99499822,
                  36.89821386,
                  36.36919498,
                  35.97099066]
        time_1 = [34.99866796,
                  35.71311212,
                  35.67857480,
                  36.37028408,
                  36.09016299,
                  36.81442618]
        time_2 = [36.23333669,
                  35.68357301,
                  35.96423864,
                  35.92800999,
                  35.83940482,
                  37.07669020]
        time_3 = [36.41870308,
                  35.75587773,
                  35.74988008,
                  36.52509904,
                  35.77912903,
                  36.73455310]
        time_4 = [36.21781015,
                  36.18444085,
                  36.19464183,
                  36.60750484,
                  36.55059600,
                  36.43489003]
        time = np.array(np.concatenate((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_kmm, error, time

    def get_data_gmm(self) -> tuple:
        """Gaussian mixture models classifier error/runtime"""
        err_0 = [0.389401,
                 0.24012,
                 0.09266600000000003,
                 0.025567000000000006,
                 0.0021090000000000275,
                 9.999999999998899e-05,
                 1.2000000000012001e-05,
                 8.000000000008e-06,
                 0.0,
                 0.0]
        err_1 = [0.37922100000000003,
                 0.232665,
                 0.10059399999999996,
                 0.026212000000000013,
                 0.0026080000000000547,
                 0.00032399999999999096,
                 4.8000000000048004e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_2 = [0.38424800000000003,
                 0.23358900000000005,
                 0.09775699999999998,
                 0.022916999999999965,
                 0.0022600000000000398,
                 0.00015900000000002024,
                 1.799999999996249e-05,
                 1.0000000000287557e-06,
                 0.0,
                 0.0]
        err_3 = [0.38414800000000004,
                 0.23629900000000004,
                 0.10221000000000002,
                 0.024878999999999984,
                 0.0022689999999999655,
                 0.00023700000000004273,
                 5.999999999950489e-06,
                 0.0,
                 0.0,
                 0.0]
        err_4 = [0.382772,
                 0.238344,
                 0.10274399999999995,
                 0.021561000000000052,
                 0.0027530000000000054,
                 0.00014499999999995072,
                 9.000000000036756e-06,
                 2.9999999999752447e-06,
                 0.0,
                 1.0000000000287557e-06]
        error = np.array(np.mean((err_0, err_1, err_2, err_3, err_4), axis=0))
        time_0 = [19.70206809,
                  19.48383117,
                  19.62756896,
                  19.75107598,
                  19.06222510,
                  19.04513407]
        time_1 = [19.06151390,
                  18.92914128,
                  19.07052231,
                  19.18112302,
                  18.98006105,
                  19.12043214]
        time_2 = [23.41747665,
                  25.66104603,
                  23.51579595,
                  23.66720104,
                  22.45848322,
                  21.04064012]
        time_3 = [21.43946123,
                  21.94604206,
                  21.35138202,
                  21.56034207,
                  24.08880472,
                  27.75989389]
        time_4 = [24.56435394,
                  23.23260021,
                  20.77032804,
                  20.09819102,
                  20.19252992,
                  19.99753094]
        time = np.array(np.concatenate((time_0, time_1, time_2, time_3, time_4), axis=0))
        return self.__name_gmm, error, time

    def get_error_supervised(self):
        """supervised error count: antenna count = 128, user count = 10, channel depth = 4"""
        powers = np.logspace(-1, 2, 10)
        powers = np.array([10 * np.log10(p_temp) for p_temp in powers])
        r_1 = self.get_data_nbc()
        r_2 = self.get_data_lrc()
        r_3 = self.get_data_dtc()
        r_4 = self.get_data_rfc()
        r_5 = self.get_data_svc()
        r_6 = self.get_data_nnc()
        n_1 = r_1[0]
        n_2 = r_2[0]
        n_3 = r_3[0]
        n_4 = r_4[0]
        n_5 = r_5[0]
        n_6 = r_6[0]
        e_1 = r_1[1]
        e_2 = r_2[1]
        e_3 = r_3[1]
        e_4 = r_4[1]
        e_5 = r_5[1]
        e_6 = r_6[1]
        if self.__show:
            print(n_1 + ' Error: ' + str(e_1))
            print(n_2 + ' Error: ' + str(e_2))
            print(n_3 + ' Error: ' + str(e_3))
            print(n_4 + ' Error: ' + str(e_4))
            print(n_5 + ' Error: ' + str(e_5))
            print(n_6 + ' Error: ' + str(e_6))
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_1[0:8], label=n_1, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_2[0:8], label=n_2, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_3[0:8], label=n_3, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_4[0:8], label=n_4, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_5[0:8], label=n_5, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_6[0:8], label=n_6, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        return

    def get_error_supervised_cmp(self):
        """supervised compare error count: antenna count = 128, user count = 10, channel depth = 4"""
        p_a = np.logspace(-1, 2, 10)
        p_a = np.array([10 * np.log10(p_temp) for p_temp in p_a])
        r_0 = self.get_data_mlp()
        r_1 = self.get_data_nbc()
        r_2 = self.get_data_lrc()
        r_3 = self.get_data_dtc()
        r_4 = self.get_data_rfc()
        r_5 = self.get_data_svc()
        r_6 = self.get_data_nnc()
        n_0 = r_0[0]
        e_0 = r_0[1]
        e_1 = r_1[1]
        e_2 = r_2[1]
        e_3 = r_3[1]
        e_4 = r_4[1]
        e_5 = r_5[1]
        e_6 = r_6[1]
        e_a = np.mean((e_1, e_2, e_3, e_4, e_5, e_6), axis=0)
        n_a = 'Supervised Learning Classifiers (Avg.)'
        if self.__show:
            print(n_0 + ' Error: ' + str(e_0))
            print(n_a + ' Error: ' + str(e_a))
        plt.xlim((p_a[0], p_a[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n$ (dB)')
        plt.semilogy(p_a[0:8], e_0[0:8], label=n_0, marker='x', c='red', mec='red', mfc='red', ms=15)
        plt.semilogy(p_a[0:8], e_a[0:8], label=n_a, marker='o', c='blue', mec='blue', mfc='None', ms=10, linestyle='--')
        plt.legend()
        plt.show()
        return

    def get_error_supervised_all(self):
        """supervised all error count: antenna count = 128, user count = 10, channel depth = 4"""
        powers = np.logspace(-1, 2, 10)
        powers = np.array([10 * np.log10(p_temp) for p_temp in powers])
        r_1 = self.get_data_nbc()
        r_2 = self.get_data_lrc()
        r_3 = self.get_data_dtc()
        r_4 = self.get_data_rfc()
        r_5 = self.get_data_svc()
        r_6 = self.get_data_nnc()
        n_1 = r_1[0]
        e_1 = r_1[1]
        n_2 = r_2[0]
        e_2 = r_2[1]
        n_3 = r_3[0]
        e_3 = r_3[1]
        n_4 = r_4[0]
        e_4 = r_4[1]
        n_5 = r_5[0]
        e_5 = r_5[1]
        n_6 = r_6[0]
        e_6 = r_6[1]
        if self.__show:
            print(n_1 + ' Error: ' + str(e_1))
            print(n_2 + ' Error: ' + str(e_2))
            print(n_3 + ' Error: ' + str(e_3))
            print(n_4 + ' Error: ' + str(e_4))
            print(n_5 + ' Error: ' + str(e_5))
            print(n_6 + ' Error: ' + str(e_6))
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_1[0:8], label=n_1, marker='s', c='b', mec='b', mfc='None', ms=15)
        plt.semilogy(powers[0:8], e_2[0:8], label=n_2, marker='o', c='g', mec='g', mfc='None', ms=15)
        plt.semilogy(powers[0:8], e_3[0:8], label=n_3, marker='d', c='r', mec='r', mfc='None', ms=10)
        plt.semilogy(powers[0:8], e_4[0:8], label=n_4, marker='*', c='purple', mec='purple', mfc='None', ms=10)
        plt.semilogy(powers[0:8], e_5[0:8], label=n_5, marker='.', c='brown', mec='brown', mfc='None', ms=5)
        plt.semilogy(powers[0:8], e_6[0:8], label=n_6, c='k')
        plt.legend()
        plt.show()
        return

    def error_unsupervised(self):
        """unsupervised error count: antenna count = 128, user count = 10, channel depth = 4"""
        powers = np.logspace(-1, 2, 10)
        powers = np.array([10 * np.log10(p_temp) for p_temp in powers])
        r_1 = self.get_data_kmm()
        r_2 = self.get_data_gmm()
        n_1 = r_1[0]
        e_1 = r_1[1]
        n_2 = r_2[0]
        e_2 = r_2[1]
        if self.__show:
            print(n_1 + ' Error: ' + str(e_1))
            print(n_2 + ' Error: ' + str(e_2))
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_1[0:8], label=n_1, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_2[0:8], label=n_2, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        return

    def error_unsupervised_cmp(self):
        """supervised compare error count: antenna count = 128, user count = 10, channel depth = 4"""
        powers = np.logspace(-1, 2, 10)
        powers = np.array([10 * np.log10(p_temp) for p_temp in powers])
        r_0 = self.get_data_mlp()
        r_1 = self.get_data_kmm()
        r_2 = self.get_data_gmm()
        n_0 = r_0[0]
        e_0 = r_0[1]
        e_1 = r_1[1]
        e_2 = r_2[1]
        e_a = np.mean((e_1, e_2), axis=0)
        n_a = 'Unsupervised Learning Classifiers (Avg.)'
        if self.__show:
            print(n_0 + ' Error: ' + str(e_0))
            print(n_a + ' Error: ' + str(e_a))
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n$ (dB)')
        plt.semilogy(powers[0:8], e_0[0:8], label=n_0, marker='x', c='red', mec='red', mfc='red', ms=15)
        plt.semilogy(powers[0:8], e_a[0:8], label=n_a, marker='o', c='blue', mec='blue', mfc='None', ms=10, linestyle='--')
        plt.legend()
        plt.show()
        return

    def error_unsupervised_all(self):
        """supervised all error count: antenna count = 128, user count = 10, channel depth = 4"""
        powers = np.logspace(-1, 2, 10)
        powers = np.array([10 * np.log10(p_temp) for p_temp in powers])
        r_1 = self.get_data_kmm()
        r_2 = self.get_data_gmm()
        n_1 = r_1[0]
        e_1 = r_1[1]
        n_2 = r_2[0]
        e_2 = r_2[1]
        if self.__show:
            print(n_1 + ' Error: ' + str(e_1))
            print(n_2 + ' Error: ' + str(e_2))
        plt.xlim((powers[0], powers[9]))
        plt.ylim((1.0e-06, 1))
        plt.ylabel('Symbol Error Rate')
        plt.xlabel('$\\rho_s$ / $\\rho_n (dB)$')
        plt.semilogy(powers[0:8], e_1[0:8], label=n_1, marker='s', c='r', mec='r', mfc='None', ms=10)
        plt.semilogy(powers[0:8], e_2[0:8], label=n_2, marker='o', c='k', mec='k', mfc='None', ms=8, linestyle='--')
        plt.legend()
        plt.show()
        return

    def runtime_supervised(self):
        """supervised runtime: antenna count = 128, user count = 10, channel depth = 4"""
        times = np.logspace(2, 7, 6, dtype='int') / np.power(10, 7)
        r_0 = self.get_data_mlp()
        r_1 = self.get_data_nbc()
        r_2 = self.get_data_lrc()
        r_3 = self.get_data_dtc()
        r_4 = self.get_data_rfc()
        r_5 = self.get_data_svc()
        r_6 = self.get_data_nnc()
        avg = np.mean(r_0[2])
        n_0 = r_0[0]
        n_1 = r_1[0]
        n_2 = r_2[0]
        n_3 = r_3[0]
        n_4 = r_4[0]
        n_5 = r_5[0]
        n_6 = r_6[0]
        t_0 = r_0[2] / avg
        t_1 = r_1[2] / avg
        t_2 = r_2[2] / avg
        t_3 = r_3[2] / avg
        t_4 = r_4[2] / avg
        t_5 = r_5[2] / avg
        t_6 = r_6[2] / avg
        if self.__show:
            print(n_0 + ' Run Time: ' + str(t_0))
            print(n_1 + ' Run Time: ' + str(t_1))
            print(n_2 + ' Run Time: ' + str(t_2))
            print(n_3 + ' Run Time: ' + str(t_3))
            print(n_4 + ' Run Time: ' + str(t_4))
            print(n_5 + ' Run Time: ' + str(t_5))
            print(n_6 + ' Run Time: ' + str(t_6))
        plt.xlim((times[0], times[-1]))
        plt.ylabel('Relative Run Time')
        plt.xlabel('Relative Training Length')
        plt.semilogx(times[0:5], t_1[0:5], label=n_1, marker='o', c='blue', mec='blue', mfc='None', ms=8)
        plt.semilogx(times[0:5], t_2[0:5], label=n_2, marker='^', c='green', mec='green', mfc='None', ms=8)
        plt.semilogx(times[0:5], t_3[0:5], label=n_3, marker='s', c='red', mec='red', mfc='None', ms=8)
        plt.semilogx(times[0:5], t_4[0:5], label=n_4, marker='*', c='black', mec='black', mfc='None', ms=10)
        plt.semilogx(times[0:5], t_5[0:5], label=n_5, marker='d', c='purple', mec='purple', mfc='None', ms=8)
        plt.legend()
        plt.show()
        return

    def runtime_supervised_cmp(self):
        """supervised compare runtime: antenna count = 128, user count = 10, channel depth = 4"""
        index = 3
        r_0 = self.get_data_mlp()
        r_1 = self.get_data_nbc()
        r_2 = self.get_data_lrc()
        r_3 = self.get_data_dtc()
        r_4 = self.get_data_rfc()
        r_5 = self.get_data_svc()
        r_6 = self.get_data_nnc()
        avg = np.mean(r_0[2])
        n_0 = r_0[0]
        n_1 = r_1[0]
        n_2 = r_2[0]
        n_3 = r_3[0]
        n_4 = r_4[0]
        n_5 = r_5[0]
        n_6 = r_6[0]
        t_0 = np.log10(r_0[2][index] / avg)
        t_1 = np.log10(r_1[2][index] / avg)
        t_2 = np.log10(r_2[2][index] / avg)
        t_3 = np.log10(r_3[2][index] / avg)
        t_4 = np.log10(r_4[2][index] / avg)
        t_5 = np.log10(r_5[2][index] / avg)
        t_6 = np.log10(r_6[2][index] / avg)
        n_a = np.array((n_0, n_1, n_2, n_3, n_4, n_5, n_6))
        t_a = np.array((t_0, t_1, t_2, t_3, t_4, t_5, t_6))
        a_a = np.linspace(0, 2 * np.pi, len(n_a), endpoint=False)
        t_a = np.concatenate((t_a, [t_a[0]]))
        a_a = np.concatenate((a_a, [a_a[0]]))
        plt.subplot(polar=True)
        plt.plot(a_a, t_a, marker='.', c='red', mec='red', mfc='red')
        plt.fill(a_a, t_a, c='red', alpha=0.25)
        plt.rgrids((0, 0.5, 1, 1.5, 2))
        plt.thetagrids(a_a * 180 / np.pi, n_a)
        plt.grid(True)
        plt.show()
        return

    def runtime_unsupervised(self):
        """unsupervised runtime: antenna count = 128, user count = 10, channel depth = 4"""
        r_0 = self.get_data_mlp()
        r_1 = self.get_data_kmm()
        r_2 = self.get_data_gmm()
        avg = np.mean(r_0[2])
        n_1 = r_1[0]
        n_2 = r_2[0]
        t_1 = r_1[2] / avg
        t_2 = r_2[2] / avg
        labels = (n_1, n_2)
        t_a = (np.mean(t_1), np.mean(t_2))
        plt.ylabel('Relative Run Time')
        plt.bar(labels, t_a, color=('red', 'blue'), width=0.5)
        plt.show()
        return

    def runtime_unsupervised_all(self):
        """unsupervised all runtime: antenna count = 128, user count = 10, channel depth = 4"""
        r_0 = self.get_data_mlp()
        r_1 = self.get_data_kmm()
        r_2 = self.get_data_gmm()
        avg = np.mean(r_0[2])
        n_0 = r_0[0]
        n_1 = r_1[0]
        n_2 = r_2[0]
        t_0 = r_0[2] / avg
        t_1 = r_1[2] / avg
        t_2 = r_2[2] / avg
        use = range(1, len(t_1) + 1)
        if self.__show:
            print(n_0 + ' Run Time: ' + str(t_0))
            print(n_1 + ' Run Time: ' + str(t_1))
            print(n_2 + ' Run Time: ' + str(t_2))
        plt.xlim((use[0], use[-1]))
        plt.ylabel('Relative Run Time')
        plt.xlabel('Channel Use')
        plt.plot(use, t_1, label=n_1, marker='o', c='red', mec='red', mfc='None', ms=8)
        plt.plot(use, t_2, label=n_2, marker='^', c='blue', mec='blue', mfc='None', ms=8)
        plt.legend()
        plt.show()
        return
