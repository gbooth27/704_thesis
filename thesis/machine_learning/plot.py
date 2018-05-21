import thesis.machine_learning.wave_function as wave
import numpy as np
import scipy as sp
import progressbar
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import thesis.machine_learning.boltzman_net as bolt


N = 10
M = N
H= 1
psi = None#wave.Psi(N, H)
psi_2 = None #wave.Psi(N, H)
h_1 = [-12.535843, -12.576526, -12.623295, -12.719091, -12.779246, -12.781792, -12.783855, -12.784371, -12.784568]
h_05 = [-10.625000, -10.625003, -10.626999, -10.633128, -10.635339, -10.635371, -10.635405, -10.635413, -10.635422]
h_2 = [-20.500383, -20.777314, -21.035684, -21.145678, -21.271187, -21.271196, -21.271197, -21.271196, -21.271204]

h_058 = [-8.500001, -8.504057, -8.506059, -8.508296, -8.508324, -8.508317, -8.508345]
h_18 = [-10.035884, -10.076317, -10.193291, -10.248026, -10.250709, -10.251404, -10.251483]
h_28 = [-16.492381, -16.712712, -16.917997, -17.018142, -17.018156, -17.018161, -17.018161]

all = [h_05, h_1, h_2]
all8 = [h_058, h_18, h_28]

if __name__ == "__main__":
    H_s = [0.5, 1, 2]
    Y = []
    Y_ACT=[]
    X = []
    for i in range(len(H_s)):
        H = H_s[i]
        bolt.tfim_builder(N)
        psi = wave.Psi(N, H)
        psi_2 = wave.Psi(N, H)

        actual = psi_2.diag()
        y_1 = [actual for _ in range(2 * N)]
        y = all[i]
        x= [j for j in range(1,N)]
        X.append(x)
        Y.append(y)
        Y_ACT.append(y_1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xlabel('Number of Hidden Units')
    ax.set_ylabel('Percent Error from True Energy')
    ax.tick_params(axis="both", which="both",  direction="in", bottom=True, top=True, left=True, right=True)
    color = ['y', 'r', 'b']
    marker = ["o", "v", "x"]
    for j in range(len(Y)):
        y_new = [((Y[j][i] - Y_ACT[j][i]) / Y_ACT[j][i]) * 100 for i in range(len(Y[j]))]
        print(y_new, H_s[j], N)
        ax.plot(X[i], np.abs(y_new), color=color[j], marker=marker[j], linestyle='solid',
                linewidth=1, markersize=6, label='h/J = '+str(H_s[j]))
    # ax.plot(x, np.log(np.abs(y_1)), "g")
    xmarks = [i for i in range(1, N, 1)]
    ax.legend()
    plt.xticks(xmarks)
    fig.savefig('../machine_learning/graphs/graph_spin_NEW_' + str(N) + '_h_' + str(H_s[i]) + '.png', bbox_inches='tight')
    fig.savefig('../machine_learning/graphs/graph_spin_NEW_' + str(N) + '_h_' + 'FINAL' + '.pdf', bbox_inches='tight')
    ######################
    fig.clear()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Y = []
    Y_ACT = []
    X = []
    N = 8
    for i in range(len(H_s)):
        H = H_s[i]
        bolt.tfim_builder(N)
        psi = wave.Psi(N, H)
        psi_2 = wave.Psi(N, H)

        actual = psi_2.diag()
        y_1 = [actual for _ in range(2 * N)]
        y = all8[i]
        x = [j for j in range(1, N)]
        X.append(x)
        Y.append(y)
        Y_ACT.append(y_1)
    ax.set_yscale('log')
    ax.set_xlabel('Number of Hidden Units')
    ax.set_ylabel('Percent Error from True Energy')
    ax.tick_params(axis="both", which="both", direction="in", bottom=True, top=True, left=True, right=True)
    color = ['y', 'r', 'b']
    marker = ["o", "v", "x"]
    for j in range(len(Y)):
        y_new = [((Y[j][i] - Y_ACT[j][i]) / Y_ACT[j][i]) * 100 for i in range(len(Y[j]))]
        print(y_new, H_s[j], N)
        ax.plot(X[i], np.abs(y_new), color=color[j], marker=marker[j], linestyle='solid',
                linewidth=1, markersize=6, label='h/J = '+str(H_s[j]))
    # ax.plot(x, np.log(np.abs(y_1)), "g")
    ax.legend()
    xmarks = [i for i in range(1, N, 1)]
    plt.xticks(xmarks)
    fig.savefig('../machine_learning/graphs/graph_spin_NEW_' + str(N) + '_h_' + str(H_s[i]) + '.png',
                bbox_inches='tight')
    fig.savefig('../machine_learning/graphs/graph_spin_NEW_' + str(N) + '_h_' + 'FINAL' + '.pdf',
                bbox_inches='tight')