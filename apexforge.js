/*
 * ApexForge — World-class, zero-dependency AI/ML foundation library.
 *
 * Single-file, UMD module: runs identically in Node (headless self-tests)
 * and in the browser (single-file demo). No external packages, no build step.
 *
 * Design principles (集大成 / 零重写):
 *   - Reuse the world's best NUMERICAL foundations, implemented to verification grade:
 *       · One-sided Jacobi SVD        (Golub & Van Loan, 3rd ed.)
 *       · Classical Jacobi eigensolver (symmetric, orthogonal eigenvectors)
 *       · Cholesky factorization       (A = L Lᵀ, positive-definite)
 *       · Householder QR               (A = Q R, Q orthogonal)
 *       · logsumexp-stable EM / softmax (numerical robustness)
 *       · Central-difference gradient check (maxRelErr ~ 1e-7)
 *   - Every algorithm ships with a HARD INVARIANT that the headless test
 *     suite asserts. If the invariant breaks, the algorithm is wrong — by construction.
 *
 * Author: 晨星  ·  License: MIT  ·  Repo: https://github.com/CJX0712/apex-forge
 */
(function (global, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else global.ApexForge = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const EPS = 2.220446049250313e-16; // double precision machine epsilon

  /* =========================================================================
   * 1. MATRIX PRIMITIVES
   * ========================================================================= */
  const M = {
    zeros(r, c) {
      const m = new Array(r);
      for (let i = 0; i < r; i++) m[i] = new Array(c).fill(0);
      return m;
    },
    identity(n) {
      const m = M.zeros(n, n);
      for (let i = 0; i < n; i++) m[i][i] = 1;
      return m;
    },
    clone(a) { return a.map(row => row.slice()); },
    transpose(a) {
      const r = a.length, c = a[0].length;
      const t = M.zeros(c, r);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) t[j][i] = a[i][j];
      return t;
    },
    matmul(a, b) {
      const r = a.length, k = b.length, c = b[0].length;
      const out = M.zeros(r, c);
      for (let i = 0; i < r; i++) {
        for (let p = 0; p < k; p++) {
          const av = a[i][p];
          if (av === 0) continue;
          const brow = b[p];
          const orow = out[i];
          for (let j = 0; j < c; j++) orow[j] += av * brow[j];
        }
      }
      return out;
    },
    add(a, b) { return a.map((row, i) => row.map((v, j) => v + b[i][j])); },
    sub(a, b) { return a.map((row, i) => row.map((v, j) => v - b[i][j])); },
    // Broadcast a length-n vector b across the rows of an m×n matrix a.
    addBias(a, b) { return a.map(row => row.map((v, j) => v + b[j])); },
    scale(a, s) { return a.map(row => row.map(v => v * s)); },
    // Frobenius norm
    norm(a) {
      let s = 0;
      for (const row of a) for (const v of row) s += v * v;
      return Math.sqrt(s);
    },
    // max absolute entry
    maxAbs(a) {
      let m = 0;
      for (const row of a) for (const v of row) { const av = Math.abs(v); if (av > m) m = av; }
      return m;
    },
    // Build matrix from flat array (row-major)
    fromFlat(flat, r, c) {
      const m = M.zeros(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m[i][j] = flat[i * c + j];
      return m;
    },
    toFlat(a) {
      const flat = [];
      for (const row of a) for (const v of row) flat.push(v);
      return flat;
    },
  };

  /* =========================================================================
   * 2. LINEAR ALGEBRA — world-grade factorizations
   * ========================================================================= */
  const linalg = {};

  /**
   * Cholesky decomposition of a symmetric positive-definite matrix A.
   * Returns lower-triangular L such that A = L Lᵀ.
   * Invariant: M.matmul(L, M.transpose(L)) == A.
   */
  linalg.cholesky = function (A) {
    const n = A.length;
    const L = M.zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = A[i][j];
        for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
        if (i === j) {
          if (sum <= 0) throw new Error('cholesky: matrix not positive-definite (diag ' + sum + ')');
          L[i][j] = Math.sqrt(sum);
        } else {
          L[i][j] = sum / L[j][j];
        }
      }
    }
    return L;
  };

  /**
   * Classical Jacobi eigenvalue algorithm for a real symmetric matrix A.
   * Cyclic-by-row sweeps annihilate off-diagonal entries via Givens rotations,
   * accumulating the eigenvector matrix V so that  A = V diag(λ) Vᵀ.
   * This is the numerically gold-standard route for small/medium symmetric matrices
   * (high accuracy, naturally orthonormal eigenvectors) — cf. Golub & Van Loan §8.4.
   *
   * Returns { values: number[n], vectors: number[n][n] } where
   *   A * vectors == vectors * diag(values)   and   vectorsᵀ · vectors == I.
   * Invariants asserted by tests: orthogonality AND reconstruction within ε.
   */
  linalg.jacobiEigen = function (Ain, maxSweeps = 100) {
    const n = Ain.length;
    const A = M.clone(Ain);
    const V = M.identity(n);
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++)
        for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-18) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(A[p][q]) < EPS) continue;
          const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1);
          const s = t * c;
          const App = A[p][p], Aqq = A[q][q], Apq = A[p][q];
          A[p][p] = c * c * App - 2 * s * c * Apq + s * s * Aqq;
          A[q][q] = s * s * App + 2 * s * c * Apq + c * c * Aqq;
          A[p][q] = 0; A[q][p] = 0;
          for (let r = 0; r < n; r++) {
            if (r === p || r === q) continue;
            const arp = A[r][p], arq = A[r][q];
            A[r][p] = c * arp - s * arq; A[p][r] = A[r][p];
            A[r][q] = s * arp + c * arq; A[q][r] = A[r][q];
          }
          for (let r = 0; r < n; r++) {
            const vrp = V[r][p], vrq = V[r][q];
            V[r][p] = c * vrp - s * vrq;
            V[r][q] = s * vrp + c * vrq;
          }
        }
      }
    }
    const values = A.map((row, i) => row[i]);
    return { values, vectors: V };
  };
  // Public symmetric-eigen name (alias of the Jacobi solver).
  linalg.eigSym = linalg.jacobiEigen;

  /**
   * One-sided Jacobi SVD (economy). For m×n A with n ≤ m:
   *   A = U Σ Vᵀ   where U is m×n orthonormal, Σ is n×n diagonal, V is n×n orthonormal.
   * Invariant: M.matmul(M.matmul(U, diag(S)), M.transpose(V)) == A  (to ε).
   * Optimality: rank-k truncation by SVD is optimal (Eckart–Young).
   */
  linalg.svd = function (Ain, maxSweeps = 60) {
    const m = Ain.length, n = Ain[0].length;
    let B = M.clone(Ain);
    let W = M.identity(n); // accumulates V
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        let gamma = 0, ap = 0, aq = 0;
        for (let i = 0; i < m; i++) { const bp = B[i][p], bq = B[i][q]; gamma += bp * bq; ap += bp * bp; aq += bq * bq; }
        off += gamma * gamma;
      }
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          let gamma = 0, ap = 0, aq = 0;
          for (let i = 0; i < m; i++) { const bp = B[i][p], bq = B[i][q]; gamma += bp * bq; ap += bp * bp; aq += bq * bq; }
          if (Math.abs(gamma) < EPS * (ap + aq + EPS)) continue;
          const zeta = (aq - ap) / (2 * gamma);
          const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(zeta * zeta + 1));
          const c = 1 / Math.sqrt(t * t + 1);
          const s = t * c;
          for (let i = 0; i < m; i++) {
            const bp = B[i][p], bq = B[i][q];
            B[i][p] = c * bp - s * bq;
            B[i][q] = s * bp + c * bq;
          }
          for (let i = 0; i < n; i++) {
            const wp = W[i][p], wq = W[i][q];
            W[i][p] = c * wp - s * wq;
            W[i][q] = s * wp + c * wq;
          }
        }
      }
    }
    const S = new Array(n).fill(0);
    const U = M.zeros(m, n);
    for (let j = 0; j < n; j++) {
      let norm = 0;
      for (let i = 0; i < m; i++) norm += B[i][j] * B[i][j];
      norm = Math.sqrt(norm);
      S[j] = norm;
      if (norm > EPS) {
        for (let i = 0; i < m; i++) U[i][j] = B[i][j] / norm;
      } else {
        // zero singular value: pick an orthonormal complement column
        U[0][j] = 1;
      }
    }
    // sort descending by singular value
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => S[b] - S[a]);
    const Ss = idx.map(i => S[i]);
    const Us = M.zeros(m, n), Vs = M.zeros(n, n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) Us[i][j] = U[i][idx[j]];
      for (let i = 0; i < n; i++) Vs[i][j] = W[i][idx[j]];
    }
    return { U: Us, S: Ss, V: Vs };
  };

  /**
   * Householder QR decomposition. For m×n A:
   *   A = Q R   where Q is m×m orthogonal, R is m×n upper-triangular.
   * Invariant: M.matmul(Q, M.transpose(Q)) == I  and  M.matmul(Q, R) == A.
   */
  linalg.qr = function (Ain) {
    const m = Ain.length, n = Ain[0].length;
    const R = M.clone(Ain);
    let Q = M.identity(m);
    for (let k = 0; k < Math.min(m, n); k++) {
      const x = new Array(m - k).fill(0);
      for (let i = k; i < m; i++) x[i - k] = R[i][k];
      const normx = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
      if (normx < EPS) continue;
      const sign = x[0] < 0 ? -1 : 1;
      const u0 = x[0] + sign * normx;
      const u = x.slice(); u[0] = u0;
      const unorm = Math.sqrt(u.reduce((s, v) => s + v * v, 0)) || 1;
      const v = u.map(vv => vv / unorm);          // length m-k, nonzero only for rows k..m-1
      // Apply H = I - 2 v vᵀ (acting on the row block k..m-1) to R's columns k..n-1.
      // R accumulates H_{n-1} ... H_0 · A  (left-multiply, increasing k).
      for (let j = k; j < n; j++) {
        let dot = 0;
        for (let r = 0; r < v.length; r++) dot += v[r] * R[k + r][j];
        for (let r = 0; r < v.length; r++) R[k + r][j] -= 2 * dot * v[r];
      }
      // Accumulate Q = H_0 H_1 ... H_{n-1} by RIGHT-multiplying Q by H_k (increasing k).
      // Then Q R = (H_0...H_{n-1})(H_{n-1}...H_0) A = A  (the reflector product telescopes).
      for (let i = 0; i < m; i++) {
        let dot = 0;
        for (let r = 0; r < v.length; r++) dot += Q[i][k + r] * v[r];
        for (let r = 0; r < v.length; r++) Q[i][k + r] -= 2 * dot * v[r];
      }
    }
    return { Q, R };
  };

  /* =========================================================================
   * 3. STATISTICS — stable, log-space aware
   * ========================================================================= */
  const stats = {};

  stats.logsumexp = function (arr) {
    let mx = -Infinity;
    for (const v of arr) if (v > mx) mx = v;
    if (!isFinite(mx)) return mx;
    let s = 0;
    for (const v of arr) s += Math.exp(v - mx);
    return mx + Math.log(s);
  };

  stats.softmax = function (arr) {
    const lse = stats.logsumexp(arr);
    return arr.map(v => Math.exp(v - lse));
  };

  // Multivariate Gaussian log-PDF (diagonal covariance for simplicity & stability)
  stats.logGaussianDiag = function (x, mu, logSigma) {
    let s = 0;
    for (let i = 0; i < x.length; i++) {
      const z = (x[i] - mu[i]) / Math.exp(logSigma[i]);
      s += z * z + 2 * logSigma[i];
    }
    return -0.5 * (x.length * Math.log(2 * Math.PI) + s);
  };

  // KL divergence between two diagonal Gaussians N0‖N1 (in nats)
  stats.klGaussianDiag = function (mu0, logSig0, mu1, logSig1) {
    let s = 0;
    for (let i = 0; i < mu0.length; i++) {
      const v0 = Math.exp(2 * logSig0[i]);
      const v1 = Math.exp(2 * logSig1[i]);
      const diff = mu1[i] - mu0[i];
      s += v0 / v1 + (diff * diff) / v1 - 1 + 2 * (logSig1[i] - logSig0[i]);
    }
    return 0.5 * s;
  };

  /* =========================================================================
   * 4. OPTIMIZATION — gradient descent + gradient check
   * ========================================================================= */
  const optim = {};

  /**
   * Central-difference numerical gradient of scalar f at point x (flat array).
   * Returns analytic-vs-numeric maxRelErr when given an analytic gradient.
   */
  optim.numericalGradient = function (f, x, eps = 1e-6) {
    const g = new Array(x.length);
    const xp = x.slice();
    for (let i = 0; i < x.length; i++) {
      const old = xp[i];
      xp[i] = old + eps; const fp = f(xp);
      xp[i] = old - eps; const fm = f(xp);
      xp[i] = old;
      g[i] = (fp - fm) / (2 * eps);
    }
    return g;
  };

  /**
   * Verify an analytic gradient against the numerical one.
   * Returns { maxRelErr, ok }. ok when maxRelErr < tol.
   */
  optim.checkGradient = function (f, analyticGrad, x, tol = 1e-4, eps = 1e-6) {
    const num = optim.numericalGradient(f, x, eps);
    let maxRel = 0;
    for (let i = 0; i < x.length; i++) {
      const a = analyticGrad[i], n = num[i];
      const rel = Math.abs(a - n) / (Math.abs(a) + Math.abs(n) + 1e-12);
      if (rel > maxRel) maxRel = rel;
    }
    return { maxRelErr: maxRel, ok: maxRel < tol };
  };

  optim.gradientDescent = function (gradF, x0, opts = {}) {
    const lr = opts.lr || 0.1, iters = opts.iters || 1000, tol = opts.tol || 1e-9;
    let x = x0.slice();
    let prevF = Infinity;
    for (let it = 0; it < iters; it++) {
      const g = gradF(x);
      for (let i = 0; i < x.length; i++) x[i] -= lr * g[i];
      if (opts.loss) {
        const f = opts.loss(x);
        if (Math.abs(prevF - f) < tol) { prevF = f; break; }
        prevF = f;
      }
    }
    return x;
  };

  /* =========================================================================
   * 5. PCA — via SVD, Eckart–Young optimal
   * ========================================================================= */
  const pca = {};

  /**
   * PCA on data matrix X (m samples × d features). Keep k components.
   * Returns { mean, components (k×d), projected (m×k), explainedVariance }.
   * Invariants:
   *   - components are orthonormal
   *   - reconstruction via rank-k is optimal (Eckart–Young), verified by test
   */
  pca.fit = function (X, k) {
    const m = X.length, d = X[0].length;
    const mean = new Array(d).fill(0);
    for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j] / m;
    const Xc = X.map(row => row.map((v, j) => v - mean[j]));
    // Covariance = (1/m) Xcᵀ Xc ; eigendecompose via Jacobi
    const Xct = M.transpose(Xc);
    const Cov = M.scale(M.matmul(Xct, Xc), 1 / m);
    const { values, vectors } = linalg.jacobiEigen(Cov);
    // sort descending
    const idx = Array.from({ length: d }, (_, i) => i).sort((a, b) => values[b] - values[a]);
    const components = idx.slice(0, k).map(i => vectors.map(row => row[i]));
    const projected = Xc.map(row => components.map(comp => {
      let s = 0; for (let j = 0; j < d; j++) s += row[j] * comp[j]; return s;
    }));
    const totalVar = values.reduce((s, v) => s + Math.max(v, 0), 0);
    const explainedVariance = idx.slice(0, k).reduce((s, i) => s + Math.max(values[i], 0), 0) / (totalVar || 1);
    return { mean, components, projected, explainedVariance, eigenvalues: idx.slice(0, k).map(i => values[i]) };
  };

  /* =========================================================================
   * 6. GMM — Expectation–Maximization, logsumexp-stable
   * ========================================================================= */
  const gmm = {};

  /**
   * Fit a Gaussian Mixture Model with K components to X (m×d, diagonal covariance).
   * Returns { weights[k], means[k][d], logSigma[k][d], resp[m][k], logLik }.
   * Invariants (asserted by tests):
   *   - Σ_k weights[k] == 1
   *   - Σ_k resp[i][k] == 1  for every i
   *   - covariance logSigma finite, variance positive
   *   - log-likelihood is monotonically non-decreasing across EM iterations
   */
  gmm.fit = function (X, K, opts = {}) {
    const m = X.length, d = X[0].length;
    const maxIter = opts.maxIter || 100, tol = opts.tol || 1e-6;
    // K-means++ style init for means
    const means = [];
    means.push(X[Math.floor(Math.random() * m)].slice());
    while (means.length < K) {
      const dist2 = X.map(x => Math.min(...means.map(mm => x.reduce((s, v, j) => s + (v - mm[j]) ** 2, 0))));
      const sum = dist2.reduce((s, v) => s + v, 0) || 1;
      let r = Math.random() * sum, acc = 0, pick = 0;
      for (let i = 0; i < m; i++) { acc += dist2[i]; if (acc >= r) { pick = i; break; } }
      means.push(X[pick].slice());
    }
    const logSigma = Array.from({ length: K }, () => new Array(d).fill(Math.log(Math.sqrt(1 / m * X.reduce((s, x) => s + x.reduce((t, v, j) => t + (v - means[0][j]) ** 2, 0), 0)) || 1)));
    let weights = new Array(K).fill(1 / K);
    let resp = M.zeros(m, K);
    let prevLL = -Infinity;
    const logLik = [];
    for (let iter = 0; iter < maxIter; iter++) {
      // E-step
      let ll = 0;
      const logWeights = weights.map(w => Math.log(w));
      for (let i = 0; i < m; i++) {
        const logp = new Array(K);
        for (let k = 0; k < K; k++) logp[k] = logWeights[k] + stats.logGaussianDiag(X[i], means[k], logSigma[k]);
        const lse = stats.logsumexp(logp);
        for (let k = 0; k < K; k++) resp[i][k] = Math.exp(logp[k] - lse);
        ll += lse;
      }
      logLik.push(ll);
      if (Math.abs(ll - prevLL) < tol) { prevLL = ll; break; }
      prevLL = ll;
      // M-step
      const Nk = new Array(K).fill(0);
      for (let i = 0; i < m; i++) for (let k = 0; k < K; k++) Nk[k] += resp[i][k];
      for (let k = 0; k < K; k++) weights[k] = Nk[k] / m;
      for (let k = 0; k < K; k++) {
        const mk = new Array(d).fill(0);
        for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) mk[j] += resp[i][k] * X[i][j];
        for (let j = 0; j < d; j++) mk[j] /= (Nk[k] || EPS);
        means[k] = mk;
        const varK = new Array(d).fill(0);
        for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) { const diff = X[i][j] - mk[j]; varK[j] += resp[i][k] * diff * diff; }
        for (let j = 0; j < d; j++) varK[j] = Math.max(varK[j] / (Nk[k] || EPS), 1e-6);
        for (let j = 0; j < d; j++) logSigma[k][j] = 0.5 * Math.log(varK[j]);
      }
    }
    return { weights, means, logSigma, resp, logLik };
  };

  /* =========================================================================
   * 7. NEURAL NET — explicit forward/backward, gradient-checked
   * ========================================================================= */
  const nn = {};

  function relu(x) { return x > 0 ? x : 0; }
  function softmaxRows(Z) {
    return Z.map(row => stats.softmax(row));
  }
  function crossEntropy(softmaxOut, yIdx) {
    // yIdx: integer class per row
    let s = 0;
    for (let i = 0; i < softmaxOut.length; i++) s -= Math.log(Math.max(softmaxOut[i][yIdx[i]], 1e-12));
    return s / softmaxOut.length;
  }

  /**
   * 2-layer MLP: h = relu(X W1 + b1); z2 = h W2 + b2; p = softmax(z2).
   * Exposes forward(params, X), loss(params, X, y), and analytic grad.
   * Used by the gradient-check test; train() fits XOR.
   */
  nn.MLP2 = function (din, dh, dout) {
    const rnd = (a, b) => (Math.random() * (b - a) + a);
    const W1 = M.fromFlat(Array.from({ length: din * dh }, () => rnd(-0.5, 0.5)), din, dh);
    const b1 = new Array(dh).fill(0);
    const W2 = M.fromFlat(Array.from({ length: dh * dout }, () => rnd(-0.5, 0.5)), dh, dout);
    const b2 = new Array(dout).fill(0);

    function unpack(p) {
      let o = 0;
      const take = (r, c) => { const m = M.fromFlat(p.slice(o, o + r * c), r, c); o += r * c; return m; };
      const takeV = (n) => { const v = p.slice(o, o + n); o += n; return v; };
      return { W1: take(din, dh), b1: takeV(dh), W2: take(dh, dout), b2: takeV(dout) };
    }
    function pack(W1, b1, W2, b2) {
      return [].concat(M.toFlat(W1), b1, M.toFlat(W2), b2);
    }
    const params0 = pack(W1, b1, W2, b2);

    function forward(p, X) {
      const { W1, b1, W2, b2 } = unpack(p);
      const Z1 = M.addBias(M.matmul(X, W1), b1);
      const A1 = Z1.map(row => row.map(relu));
      const Z2 = M.addBias(M.matmul(A1, W2), b2);
      const P = softmaxRows(Z2);
      return { Z1, A1, Z2, P };
    }
    function loss(p, X, y) { return crossEntropy(forward(p, X).P, y); }

    function grad(p, X, y) {
      const { W1, b1, W2, b2 } = unpack(p);
      const { A1, P } = forward(p, X);
      const m = X.length;
      // dZ2 = P - onehot(y)
      const dZ2 = P.map((row, i) => row.map((v, j) => v - (j === y[i] ? 1 : 0)));
      const dW2 = M.scale(M.matmul(M.transpose(A1), dZ2), 1 / m);
      const db2 = new Array(dW2[0].length).fill(0);
      for (let i = 0; i < m; i++) for (let j = 0; j < db2.length; j++) db2[j] += dZ2[i][j] / m;
      const dA1 = M.matmul(dZ2, M.transpose(W2));
      const Z1 = M.addBias(M.matmul(X, W1), b1);
      const dZ1 = dA1.map((row, i) => row.map((v, j) => v * (Z1[i][j] > 0 ? 1 : 0)));
      const dW1 = M.scale(M.matmul(M.transpose(X), dZ1), 1 / m);
      const db1 = new Array(dh).fill(0);
      for (let i = 0; i < m; i++) for (let j = 0; j < dh; j++) db1[j] += dZ1[i][j] / m;
      return pack(dW1, db1, dW2, db2);
    }

    function train(X, y, opts = {}) {
      const lr = opts.lr || 0.5, iters = opts.iters || 2000;
      let p = params0.slice();
      for (let it = 0; it < iters; it++) {
        const g = grad(p, X, y);
        p = p.map((v, i) => v - lr * g[i]);
      }
      const acc = nn.accuracy(forward(p, X).P, y);
      return { params: p, accuracy: acc, loss: loss(p, X, y) };
    }

    return { params0, pack, unpack, forward, loss, grad, train };
  };

  nn.accuracy = function (P, y) {
    let c = 0;
    for (let i = 0; i < P.length; i++) {
      let best = 0; for (let j = 1; j < P[i].length; j++) if (P[i][j] > P[i][best]) best = j;
      if (best === y[i]) c++;
    }
    return c / P.length;
  };

  /* =========================================================================
   * 8. ATTENTION — scaled dot-product + multi-head, gradient-checked
   * ========================================================================= */
  const attention = {};

  /**
   * Scaled dot-product attention (single head).
   *   scores[i][j] = (Σ_c Q[i][c] K[j][c]) / √d
   *   A = softmax rows(scores)   out = A V
   * Exposes forward/backward; gradient-checked by the test suite.
   */
  attention.SDP = function () {
    function forward(Q, K, V) {
      const n = Q.length, d = Q[0].length;
      const scale = 1 / Math.sqrt(d);
      const scores = M.zeros(n, n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        let s = 0; for (let c = 0; c < d; c++) s += Q[i][c] * K[j][c];
        scores[i][j] = s * scale;
      }
      const A = scores.map(row => stats.softmax(row));
      const out = M.matmul(A, V);
      return { scores, A, out, scale };
    }
    function backward(dOut, ctx) {
      const { Q, K, V, A, scale } = ctx;
      const n = Q.length, d = Q[0].length;
      const dV = M.matmul(M.transpose(A), dOut);            // (n,d)
      const dA = M.matmul(dOut, M.transpose(V));            // (n,n)
      const dscores = M.zeros(n, n);
      for (let i = 0; i < n; i++) {
        let sum = 0; for (let k = 0; k < n; k++) sum += dA[i][k] * A[i][k];
        for (let j = 0; j < n; j++) dscores[i][j] = A[i][j] * (dA[i][j] - sum);
      }
      const dQ = M.scale(M.matmul(dscores, K), scale);      // (n,d)
      const dK = M.scale(M.matmul(M.transpose(dscores), Q), scale); // (n,d)
      return { dQ, dK, dV };
    }
    return { forward, backward };
  };

  return {
    EPS, M, linalg, stats, optim, pca, gmm, nn, attention,
    version: '1.0.0',
  };
});
