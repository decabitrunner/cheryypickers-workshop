import bisect
from flask import Flask, render_template, jsonify
import numpy as np
from scipy import stats

app = Flask(__name__)


# ── Longest Increasing Subsequence (O n log n) ──────────────────────────────

def lis_indices(arr):
    """Return the indices of one LIS in arr using patience sorting."""
    n = len(arr)
    if n == 0:
        return []

    tails = []       # tails[k] = smallest tail element of any IS of length k+1
    tail_pos = []    # tail_pos[k] = index in arr of tails[k]
    pred = [-1] * n  # pred[i] = predecessor index of arr[i] in its IS

    for i, val in enumerate(arr):
        pos = bisect.bisect_left(tails, val)
        if pos == len(tails):
            tails.append(val)
            tail_pos.append(i)
        else:
            tails[pos] = val
            tail_pos[pos] = i
        pred[i] = tail_pos[pos - 1] if pos > 0 else -1

    # Reconstruct path
    seq = []
    idx = tail_pos[-1]
    while idx != -1:
        seq.append(idx)
        idx = pred[idx]
    seq.reverse()
    return seq


def best_monotone_subset(points):
    """
    Sort points by x, then find the longest monotone subsequence in y
    (either increasing or decreasing). Return global indices and Pearson r.
    """
    n = len(points)
    order = sorted(range(n), key=lambda i: points[i][0])
    y = [points[order[i]][1] for i in range(n)]

    lis_local = lis_indices(y)
    lds_local = lis_indices([-v for v in y])   # LDS = LIS of negated values

    lis_global = [order[i] for i in lis_local]
    lds_global = [order[i] for i in lds_local]

    def pearson_r(idx_list):
        if len(idx_list) < 3:
            return 0.0
        xs = [points[i][0] for i in idx_list]
        ys = [points[i][1] for i in idx_list]
        return float(stats.pearsonr(xs, ys)[0])

    r_inc = pearson_r(lis_global)
    r_dec = pearson_r(lds_global)

    if abs(r_inc) >= abs(r_dec):
        return lis_global, r_inc
    else:
        return lds_global, r_dec


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate')
def generate():
    n = 22
    rng = np.random.default_rng()
    xs = rng.uniform(0, 100, n).round(1).tolist()
    ys = rng.uniform(0, 100, n).round(1).tolist()
    pts = list(zip(xs, ys))

    subset, r_sub = best_monotone_subset(pts)
    ignored = [i for i in range(n) if i not in subset]

    r_all = float(stats.pearsonr(xs, ys)[0])

    sx = [pts[i][0] for i in subset]
    sy = [pts[i][1] for i in subset]
    m_s, b_s = np.polyfit(sx, sy, 1)
    m_a, b_a = np.polyfit(xs, ys, 1)

    deception = round(r_sub ** 2 / r_all ** 2, 1) if abs(r_all) > 0.01 else None

    return jsonify({
        'points':    [[p[0], p[1]] for p in pts],
        'subset':    subset,
        'ignored':   ignored,
        'r_sub':     round(r_sub, 3),
        'r_all':     round(r_all, 3),
        'r2_sub':    round(r_sub ** 2, 3),
        'r2_all':    round(r_all ** 2, 3),
        'n_sub':     len(subset),
        'n_total':   n,
        'reg_sub':   [round(float(m_s), 4), round(float(b_s), 4)],
        'reg_all':   [round(float(m_a), 4), round(float(b_a), 4)],
        'deception': deception,
    })


if __name__ == '__main__':
    app.run(debug=True)
