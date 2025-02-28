from collections import Counter
from pathlib import Path

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr, erfc
from scipy.stats import entropy, linregress, pearsonr
import seaborn as sns

PATH_BASE = Path("/home/jmcbride/projects/PelofiReplyCode")



####################################################################
### The main code used to generate "JSD_vs_dprime.pdf"


# Load Markov model transition matrix
# The "inv" argument changes the grammars from `correct' (inv=False) to `incorrect' in Exp 2 and 3
def load_mat(path, inv=False, norm=False, inv_key=None):
    i, j, p = np.loadtxt(path).T
    i = i.astype(int)
    j = j.astype(int)
    n = np.max(np.append(i,j))
    mat = np.zeros((n,n), float)
    if isinstance(inv_key, type(None)):
        inv_key = {3:4, 4:3, 5:6, 6:5}

    for k in range(i.size):
        if inv:
            l = inv_key.get(i[k], i[k]) - 1
            m = inv_key.get(j[k], j[k]) - 1
        else:
            l, m = i[k] - 1, j[k] - 1
        mat[l,m] = p[k]
    if norm:
        for i, m in enumerate(mat):
            if np.sum(m) > 0:
                mat[i] = m / m.sum()
    return mat


# Load possible transitions and probabilities
def load_grammar(n, inv=False):
    return load_mat(PATH_BASE.joinpath(f'Grammars/transitions_{n}.txt'), inv)


# Load Markov model transition matrix foor Exp 1
def load_grammar_exp1():
    pairs = np.loadtxt(PATH_BASE.joinpath('Grammars/exp1.txt'), int)
    mat = np.zeros((8,8), float)
    np.fill_diagonal(mat, 1)
    for i, j in zip(*pairs.T):
        mat[i, j] = 1
        mat[j, i] = 1
    for i in range(8):
        mat[i] = mat[i] / np.sum(mat[i])
    return mat


# Generate melodies according to Exp 1
def generate_melodies_exp1(n_mel=10000, mel_len=15, inv=False):
    mat = load_grammar_exp1()
    # Key for converting markov model indices to notes
    key_path = PATH_BASE.joinpath('Grammars/exp1_map.txt')
    mel_key = {i:j for i, j in zip(*np.loadtxt(key_path, dtype=int, delimiter=',', skiprows=1).T)}
    melodies = []
    for i in range(n_mel):
        if not inv:
            m = generate_melody(mat, mel_len, 0)
        else:
            m = generate_melody(mat, mel_len, 7)
        melodies.append([mel_key[i] for i in m])
    return melodies


# Data is given in order - Uni, Sym, Asym -
# so need to reverse the array
def load_dprime(exp, n):
    return np.loadtxt(PATH_BASE.joinpath(f'dprime/exp{exp}_{n}.txt'))[::-1]


# Standard error
def load_dprime_error(exp, n):
    dprime = np.loadtxt(PATH_BASE.joinpath(f'dprime/exp{exp}_{n}.txt'))[::-1]
    dprime_plus_error = np.loadtxt(PATH_BASE.joinpath(f'dprime/exp_{exp}_{n}_err.txt'))[::-1]
    return dprime_plus_error - dprime


def load_dprime_recognition():
    dprime = np.loadtxt(PATH_BASE.joinpath(f'dprime/exp1_rec.txt'))[::-1,1]
    dprime_plus_error = np.loadtxt(PATH_BASE.joinpath(f'dprime/exp1_rec_err.txt'))[::-1,1]
    return dprime, dprime_plus_error - dprime


def exp1_scales():
    scale_uni = np.array([0, 2, 4, 6, 8, 10, 12])
    scale_sym = np.array([0, 1, 4, 6, 7, 10, 12])
    scale_asym = np.array([0, 2, 4, 6, 9, 10, 12])
    return scale_uni, scale_sym, scale_asym


def exp2_scales(n=6):
    if n == 6:
        scale_uni = np.array([0, 2, 4, 6, 8, 10, 12])
        scale_sym = np.array([0, 2, 3, 6, 9, 10, 12])
        scale_asym = np.array([0, 1, 3, 4, 7, 10, 12])
    elif n == 7:
        scale_uni = np.array([0, 2, 4, 6, 8, 10, 12, 14])
        scale_sym = np.array([0, 2, 3, 6, 8, 11, 12, 14])
        scale_asym = np.array([0, 1, 3, 5, 8, 9, 11, 14])
    return scale_uni, scale_sym, scale_asym


def exp3_scales(n=6):
    if n == 6:
        scale_uni = np.array([0, 2, 4, 6, 8, 10, 12])
        scale_sym = np.array([0, 1, 4, 6, 7, 10, 12])
        scale_asym = np.array([0, 1, 3, 4, 7, 10, 12])
    elif n == 8:
        scale_uni = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])
        scale_sym = np.array([0, 1, 4, 6, 8, 9, 12, 14, 16])
        scale_asym = np.array([0, 1, 4, 6, 9, 11, 13, 14, 16])
    return scale_uni, scale_sym, scale_asym


# Generate n melodies using a transition matrix
# starting on note with index i
def generate_melody(mat, n, i=0):
    mel = [i]
    for j in range(1, n):
        i = np.random.choice(range(len(mat)), p=mat[i])
        mel.append(i)
        if i == len(mat) - 1:
            break
    return np.array(mel)


# Convert melodies (in the form of scale degrees) to
# intervals (in units of `semitones', or whatever the equivalent unit
# is in 14-TET and 16-TET)
def convert_mel_to_int(mel, scale):
    notes = scale[mel]
    return np.diff(notes)


# Get entropy of a list of sequences
def get_entropy(sequences):
    return entropy(list(Counter([x for y in sequences for x in y]).values()))


# Get Jensen-Shannon Divergence of two lists of sequences
def get_JSD(s1, s2):
    c1 = Counter([x for y in s1 for x in y])
    n1 = np.sum(list(c1.values()))
    c2 = Counter(x for y in s2 for x in y)
    n2 = np.sum(list(c2.values()))
    keys = sorted(set(c1.keys()).union(c2.keys()))
    p1 = [c1.get(k,0)/n1 for k in keys]
    p2 = [c2.get(k,0)/n2 for k in keys]
    return jensenshannon(p1, p2)


# Get KL Divergence of two lists of sequences
def get_kl_divergence(s1, s2):
    c1 = Counter([x for y in s1 for x in y])
    n1 = np.sum(list(c1.values()))
    c2 = Counter(x for y in s2 for x in y)
    n2 = np.sum(list(c2.values()))
    p1 = [c1.get(k,0)/n1 for k in c2.keys()]
    p2 = [c2.get(k,0)/n2 for k in c2.keys()]
    print(p1, p2)
    return np.sum(rel_entr(p1, p2))


# Run a simulation of experiment (1, 2 or 3)
# Returns the entropy of the `correct' and `incorrect' grammars, and the JSD between them
# n: number of notes in a scale
# exp: experiment number
# mel_len: maximum length of generated melody
# nmel: number of melodies to generate
def run_exp(n=6, exp=2, mel_len=15, nmel=10000):

    if exp == 1:
        melodies = generate_melodies_exp1(nmel, mel_len, False)
        melodies_inv = generate_melodies_exp1(nmel, mel_len, True)
    else:
        mat = load_grammar(n*2)
        mat_inv = load_grammar(n*2, True)
        melodies = [generate_melody(mat, mel_len) for i in range(nmel)]
        melodies_inv = [generate_melody(mat_inv, mel_len) for i in range(nmel)]

    if exp == 1:
        scales = exp1_scales()
    elif exp == 2:
        scales = exp2_scales(n)
    elif exp ==3:
        scales = exp3_scales(n)

    out = []

    for scale in scales:
        int_seq = [convert_mel_to_int(m, scale) for m in melodies]
        int_seq_inv = [convert_mel_to_int(m, scale) for m in melodies_inv]
        ent = get_entropy(int_seq)
        ent_inv = get_entropy(int_seq_inv)
        jsd = get_JSD(int_seq, int_seq_inv)
        out.append([ent, ent_inv, jsd])

    return np.array(out)


def plot_JSD_vs_dprime():
    out = []
    dprime = []
    dprime_err = []
    lbls = []
    for i, exp in zip([1,2], [2,3]):
        if exp == 1:
            n = 6
            out.append(run_exp(n, exp))
            dprime.extend(list(load_dprime(exp, n*2)))
            dprime_err.extend(list(load_dprime_error(exp, n*2)))
            lbls.extend([(exp,n,a) for a in ['Uni', 'Sym', 'Asym']])
        else:
            for j, n in enumerate([(6, 7), (6, 8)][i-1]):
                out.append(run_exp(n, exp))
                dprime.extend(list(load_dprime(exp, n*2)))
                dprime_err.extend(list(load_dprime_error(exp, n*2)))
                lbls.extend([(exp,n,a) for a in ['Uni', 'Sym', 'Asym']])
    ent, ent_inv, jsd = np.concatenate(out).T
    sns.regplot(x=list(jsd), y=np.array(dprime)[:,1])
    return out, dprime, dprime_err, lbls


def get_exp1_entropy():
    path = Path('../Simulation/exp1.npy')
    if path.exists():
        res = np.load(path)
    else:
        res = np.array([run_exp(6, 1, nmel=100)[:,0] for i in range(10000)])
        np.save(path, res)
    return res.mean(axis=0), res.std(axis=0)


def annotate_stats(ax, x0, x1, y0, y1, txt):
    ax.plot([x0, x0, x1, x1], [y0, y1, y1, y0], '-k', lw=1.5)
    x2 = np.mean([x0, x1])
    dy = (y1 - y0)
    ax.plot([x2, x2], [y1, y1 + dy], '-k', lw=1.5)
    ax.text(x2 - len(txt)*.5 * 0.03, y1 + dy*2, txt)


def JSD_vs_dprime_paper_figure(jsd, dprime, dprime_err, lbls):
    exp = np.array(lbls)[:,0].astype(int)
    n = np.array(lbls)[:,1].astype(int)
    scale = np.array(lbls)[:,2]
    dprime = np.array(dprime)[:,1]
    dprime_err = np.array(dprime_err)[:,1]
    df = pd.DataFrame({'exp':exp, 'n':n, 'sym':scale, 'dprime':dprime,
                       'dprime_err':dprime_err, 'jsd':jsd})

    exp1_entropy, exp1_std = get_exp1_entropy()

    fig = plt.figure(figsize=(9,9))
    gs = GridSpec(2,1, height_ratios=[1,2.5])
    ax = [fig.add_subplot(g) for g in gs]
    fig.subplots_adjust(hspace=0.4)
    col = sns.color_palette()

    ax[0].bar(range(3), exp1_entropy, 0.5, yerr=exp1_std, color=col[:3])

    p1 = 0.5 * erfc((exp1_entropy[1] - exp1_entropy[0]) / (2 * (exp1_std[1]**2 + exp1_std[0]**2))**0.5)
    p2 = 0.5 * erfc((exp1_entropy[2] - exp1_entropy[1]) / (2 * (exp1_std[2]**2 + exp1_std[1]**2))**0.5)
    y1 = exp1_entropy[1] + 0.2
    y2 = exp1_entropy[2] + 0.3
    annotate_stats(ax[0], 0, 1, y1, y1+0.1, f"p = {p1:5.3e}")
    annotate_stats(ax[0], 1, 2, y2, y2+0.1, f"p = {p2:5.3f}")


    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(['Uniform', 'Symmetric', 'Asymmetric'])
    ax[0].set_xlabel("Scale used to generate melodies in Experiment 1")
    ax[0].set_ylabel("Entropy of interval distribution of melodies\ngenerated using Grammar group 1")
    ax[0].set_ylim(0,3.2)
    ax[0].set_yticks(np.arange(0,3.1,1))

    for s, l, c in zip(['Uni', 'Sym', 'Asym'], ['Uniform', 'Symmetric', 'Asymmetric'], col):
        for e, p in zip([1,2,3], '^os'):
            X, Y, Yerr = df.loc[(df.exp==e)&(df.sym==s), ['jsd', 'dprime', 'dprime_err']].values.T
            ax[1].errorbar(X, Y, yerr=Yerr, fmt=p, color=c, ms=8, mec='k', mew=2)
    sns.regplot(x='jsd', y='dprime', data=df, scatter=False, ax=ax[1])
    ax[1].set_xlabel("Jensen-Shannon divergence between interval distributions\ndrawn from different grammars")
    ax[1].set_ylabel("d'")
    ax[1].set_ylim(0, 2.1)
    ax[1].set_yticks(np.arange(0,2.1,.5))
    ax[1].set_xticks(np.arange(0.3, 0.7, 0.1))

    handles = [Line2D([0], [0], marker=p, color='k', fillstyle='none', ms=8, mew=2) for p in 'os'] + \
              [Patch(facecolor='white')] + [Patch(facecolor=c) for c in col]
    labels = ['Experiment 2', 'Experiment 3', '', 'Uniform', 'Symmetric', 'Asymmetric']
    ax[1].legend(handles, labels, loc='lower right', frameon=False, ncol=2)

    r, p = pearsonr(list(jsd), dprime)
    ax[1].text(0.7, 0.43, f"r = {r:4.2f}", transform=ax[1].transAxes)
    ax[1].text(0.7, 0.36, f"p = {p:5.3f}", transform=ax[1].transAxes)

    ax[0].text(-0.13, 1.15, 'A', transform=ax[0].transAxes, fontsize=18)
    ax[1].text(-0.13, 1.03, 'B', transform=ax[1].transAxes, fontsize=18)

    for a in ax:
        for d in ['top', 'right']:
            a.spines[d].set_visible(False)

    fig.savefig("../JSD_vs_dprime.pdf", bbox_inches='tight')



####################################################################
### Code used to test information properties of alternate grammars


def invert_grammar(mat):
    mat = mat.copy()
    inv_key = {2:3, 4:5}
    for i, j in inv_key.items():
        a, b = mat[:,i].copy(), mat[:,j].copy()
        mat[:,i], mat[:,j] = b, a
        a, b = mat[i,:].copy(), mat[j,:].copy()
        mat[i,:], mat[j,:] = b, a
    return mat


def random_grammar_prob(mat):
    mat = np.zeros_like(mat)
    for i in range(len(mat)):
#       idx = np.where(mat[i]!=0.)[0]
        idx = np.arange(i + 1, len(mat))
        prob = np.random.rand(len(idx))
        prob = prob / prob.sum()
        mat[i,idx] = prob
    return mat


def uniform_grammar_prob(mat):
    mat = np.zeros_like(mat)
    for i in range(len(mat)):
        idx = np.arange(i + 1, len(mat))
        mat[i,idx] = np.ones(len(idx)) / len(idx)
    return mat


def alternative_grammars(exp, n, nmel=10000, nmat=20):
    mat = load_grammar(n*2)
    if exp == 2:
        scales = exp2_scales(n)
    else:
        scales = exp3_scales(n)
    fig, ax = plt.subplots()
    for scale, lbl in zip(scales, ['Uniform', 'Symmetric', 'Asymmetric']):
        melodies, melodies_inv = [], []
        for j in range(nmat):
            ran_mat = random_grammar_prob(mat)
            ran_mat_inv = random_grammar_prob(mat)
#           ran_mat_inv = uniform_grammar_prob(mat)
#           ran_mat_inv = invert_grammar(ran_mat)
            melodies.append([generate_melody(ran_mat, 15) for i in range(nmel)])
            melodies_inv.append([generate_melody(ran_mat_inv, 15) for i in range(nmel)])
        int_seq = [[convert_mel_to_int(m, scale) for m in mel] for mel in melodies]
        int_seq_inv = [[convert_mel_to_int(m, scale) for m in mel] for mel in melodies_inv]

        jsd = [get_JSD(int_seq[i], int_seq_inv[i]) for i in range(nmat)]
        sns.distplot(jsd, label=lbl)
    ax.legend(loc='best', frameon=False)


####################################################################
### Comparing contours in Pelofi & Farbood


# Run a simulation of experiment (1, 2 or 3)
# Returns the entropy of the `correct' and `incorrect' grammars, and the JSD between them
# n: number of notes in a scale
# exp: experiment number
# mel_len: maximum length of generated melody
# nmel: number of melodies to generate
def run_exp_contour(n=6, exp=2, mel_len=15, nmel=10000):

    # Generate melodies using grammar from Pelofi & Farbood
    if exp == 1:
        melodies = generate_melodies_exp1(nmel, mel_len, False)
        melodies_inv = generate_melodies_exp1(nmel, mel_len, True)
    else:
        mat = load_grammar(n*2)
        mat_inv = load_grammar(n*2, True)
        melodies = [generate_melody(mat, mel_len) for i in range(nmel)]
        melodies_inv = [generate_melody(mat_inv, mel_len) for i in range(nmel)]

    # Convert melodies to contours
    key = {1:'+', -1:'-', 0:'_'}
    contours = np.array([''.join(key[x] for x in np.sign(np.diff(m))) for m in melodies])
    contours_inv = np.array([''.join(key[x] for x in np.sign(np.diff(m))) for m in melodies_inv])

    # Get contour distributions
    total_set = np.array(list(set(contours).union(set(contours_inv))))
    fraction = np.array([np.mean(contours == c) for c in total_set])
    fraction_inv = np.array([np.mean(contours_inv == c) for c in total_set])

    # Plot contour distributions
    order = np.argsort(fraction)[::-1]
    fig, ax = plt.subplots()
    width = 0.3
    X = np.arange(len(total_set))
    ax.bar(X - width / 2, fraction[order], width, ec='k')
    ax.bar(X + width / 2, fraction_inv[order], width, ec='k')

    threshold = 0.03
    xticklabels = [c if (f1 > threshold) | (f2 > threshold) else ''
                   for c, f1, f2 in zip(total_set, fraction[order], fraction_inv[order])]
    ax.set_xticks(X)
    ax.set_xticklabels(xticklabels, rotation=90)

    JSD = jensenshannon(fraction, fraction_inv)
    print(f"Jensen Shannon divergence between contour distributions: {JSD:5.2f}")

    







