/*
 * ApexForge — headless self-test suite (zero dependency).
 * Asserts every HARD INVARIANT of every algorithm. Run: node test/run.js
 * Exit code 0 = all green, 1 = at least one invariant violated.
 */
const AF = require('../apexforge.js');
const { M, linalg, stats, optim, pca, gmm, nn, attention } = AF;

let pass = 0, fail = 0;
const fails = [];
function check(name, cond, detail = '') {
  if (cond) { pass++; console.log('  ✅ ' + name); }
  else { fail++; fails.push(name); console.log('  ❌ ' + name + (detail ? '  — ' + detail : '')); }
}
function close(a, b, tol = 1e-8) { return Math.abs(a - b) <= tol; }
function matClose(A, B, tol = 1e-7) {
  if (!A || !B || A.length !== B.length || A[0].length !== B[0].length) return false;
  let mx = 0;
  for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) mx = Math.max(mx, Math.abs(A[i][j] - B[i][j]));
  return mx <= tol;
}
function isOrthogonal(Mm, tol = 1e-7) {
  const I = M.matmul(Mm, M.transpose(Mm));
  return matClose(I, M.identity(Mm.length), tol);
}

// deterministic RNG for reproducibility
let _seed = 123456789;
function rnd() { _seed = (_seed * 1103515245 + 12345) & 0x7fffffff; return _seed / 0x7fffffff; }
function rndMat(r, c, lo = -1, hi = 1) { return M.fromFlat(Array.from({ length: r * c }, () => lo + rnd() * (hi - lo)), r, c); }

console.log('\n=== ApexForge self-test ===');

/* 1. Cholesky */
console.log('\n[1] Cholesky factorization');
{
  const B0 = rndMat(5, 5);
  const A = M.matmul(B0, M.transpose(B0));
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) A[i][j] += (i === j ? 5 : 0); // SPD
  const L = linalg.cholesky(A);
  const recon = M.matmul(L, M.transpose(L));
  check('A = L Lᵀ (reconstruction)', matClose(recon, A, 1e-7));
  check('L is lower triangular', L.every((row, i) => row.slice(i + 1).every(v => v === 0)));
}

/* 2. Jacobi eigen */
console.log('\n[2] Jacobi eigenvalue decomposition');
{
  const B0 = rndMat(6, 6);
  const A = M.matmul(B0, M.transpose(B0));
  const { values, vectors } = linalg.jacobiEigen(A);
  check('eigenvectors orthogonal', isOrthogonal(vectors, 1e-7));
  const AV = M.matmul(A, vectors);
  // V Λ: scale COLUMN j of V by values[j] (eigenvectors are stored as columns of V)
  const VD = vectors.map(row => row.map((v, j) => v * values[j]));
  check('A V = V Λ (reconstruction)', matClose(AV, VD, 1e-6));
}

/* 3. SVD */
console.log('\n[3] One-sided Jacobi SVD');
{
  const A = rndMat(7, 4, -2, 2);
  const n = A[0].length;
  const { U, S, V } = linalg.svd(A);
  // reconstruct A = U diag(S) Vᵀ
  const US = U.map(row => row.map((v, j) => v * S[j]));
  const recon = M.matmul(US, M.transpose(V));
  check('A = U Σ Vᵀ (reconstruction)', matClose(recon, A, 1e-6));
  check('U columns orthonormal (UᵀU = I)', matClose(M.matmul(M.transpose(U), U), M.identity(n), 1e-7));
  check('V is orthonormal', isOrthogonal(V, 1e-7));
  // Eckart–Young: rank-k SVD reconstruction is optimal.
  // Compare against a NON-optimal rank-3 basis (first 3 cols of a random orthogonal matrix).
  const k = 2;
  const fullUSVt = M.matmul(US, M.transpose(V));
  // rank-k truncation: keep first k singular values
  const Sk = S.map((s, j) => (j < k ? s : 0));
  const USk = U.map(row => row.map((v, j) => v * Sk[j]));
  const Ak = M.matmul(USk, M.transpose(V));
  const errOpt = M.norm(M.sub(A, Ak));
  // non-optimal basis: random orthogonal Q (via QR of random), keep first k columns
  const qrR = linalg.qr(rndMat(7, 4)) ; const Q = qrR.Q;
  const Qk = Q.map(row => row.slice(0, k));
  const proj = M.matmul(M.matmul(Qk, M.transpose(Qk)), A); // rank-k projection (non-optimal)
  const errRand = M.norm(M.sub(A, proj));
  check('Eckart–Young: rank-k SVD ≤ random rank-k (optimal)', errOpt <= errRand + 1e-6, `errOpt=${errOpt.toExponential(3)} errRand=${errRand.toExponential(3)}`);
}

/* 4. QR */
console.log('\n[4] Householder QR');
{
  const A = rndMat(6, 4, -2, 2);
  const { Q, R } = linalg.qr(A);
  check('Q is orthogonal', isOrthogonal(Q, 1e-7));
  const recon = M.matmul(Q, R);
  check('A = Q R (reconstruction)', matClose(recon, A, 1e-6));
  check('R is upper triangular', R.every((row, i) => row.slice(0, i).every(v => Math.abs(v) < 1e-9)));
}

/* 5. Statistics */
console.log('\n[5] Statistics (stable)');
{
  const sm = stats.softmax([1, 2, 3]);
  check('softmax sums to 1', close(sm.reduce((s, v) => s + v, 0), 1, 1e-12));
  // logsumexp identity: log(sum exp(x)) 
  const x = [0.5, -1.2, 3.1, -0.4];
  const lse = stats.logsumexp(x);
  const ref = Math.log(x.reduce((s, v) => s + Math.exp(v), 0));
  check('logsumexp correct', close(lse, ref, 1e-9), `lse=${lse} ref=${ref}`);
  // KL ≥ 0
  const mu0 = [0, 0], ls0 = [0, 0], mu1 = [0.3, -0.2], ls1 = [0.1, 0.2];
  const kl = stats.klGaussianDiag(mu0, ls0, mu1, ls1);
  check('KL divergence ≥ 0', kl >= -1e-12, `KL=${kl}`);
  const klSelf = stats.klGaussianDiag(mu0, ls0, mu0, ls0);
  check('KL(p‖p) = 0', close(klSelf, 0, 1e-9));
}

/* 6. PCA */
console.log('\n[6] PCA (SVD-based)');
{
  // correlated 2D data
  const X = [];
  for (let i = 0; i < 50; i++) { const t = rnd(); X.push([t * 3 + rnd() * 0.1, t * 3 + rnd() * 0.1]); }
  const k = 1;
  const res = pca.fit(X, k);
  // components orthonormal
  const comp = res.components; // k×d
  let ortho = true;
  for (let a = 0; a < comp.length; a++) for (let b = a + 1; b < comp.length; b++) {
    let dot = 0; for (let j = 0; j < comp[a].length; j++) dot += comp[a][j] * comp[b][j];
    if (Math.abs(dot) > 1e-7) ortho = false;
  }
  check('PCA components orthonormal', ortho);
  check('PCA explained variance in (0,1]', res.explainedVariance > 0 && res.explainedVariance <= 1 + 1e-9, `ev=${res.explainedVariance}`);
  check('PCA recovers dominant direction', Math.abs(Math.abs(res.components[0][0]) - Math.abs(res.components[0][1])) < 0.2, `comp=${res.components[0]}`);
}

/* 7. GMM */
console.log('\n[7] GMM via EM');
{
  // two well-separated clusters
  const X = [];
  for (let i = 0; i < 60; i++) X.push([rnd() * 0.3, rnd() * 0.3]);
  for (let i = 0; i < 60; i++) X.push([5 + rnd() * 0.3, 5 + rnd() * 0.3]);
  const model = gmm.fit(X, 2, { maxIter: 80 });
  check('GMM weights sum to 1', close(model.weights.reduce((s, v) => s + v, 0), 1, 1e-9));
  const respOK = model.resp.every(row => close(row.reduce((s, v) => s + v, 0), 1, 1e-9));
  check('GMM responsibilities sum to 1 per sample', respOK);
  // log-likelihood monotonic non-decreasing
  let mono = true;
  for (let i = 1; i < model.logLik.length; i++) if (model.logLik[i] < model.logLik[i - 1] - 1e-6) mono = false;
  check('GMM log-likelihood non-decreasing', mono, `nll0=${model.logLik[0].toFixed(2)} nllN=${model.logLik[model.logLik.length-1].toFixed(2)}`);
}

/* 8. Neural net — gradient check + XOR */
console.log('\n[8] Neural net (MLP2)');
{
  const din = 3, dh = 4, dout = 2;
  const mlp = nn.MLP2(din, dh, dout);
  const Xs = rndMat(6, din, -1, 1);
  const ys = [0, 1, 0, 1, 0, 1];
  const lossF = (p) => mlp.loss(p, Xs, ys);
  const anaG = mlp.grad(mlp.params0, Xs, ys);
  const { maxRelErr, ok } = optim.checkGradient(lossF, anaG, mlp.params0, 1e-4, 1e-6);
  check('MLP analytic grad matches numerical', ok, `maxRelErr=${maxRelErr.toExponential(3)}`);

  // XOR solve via restart loop (deterministic seed)
  let solved = false, acc = 0, loss = 0;
  for (let t = 0; t < 40 && !solved; t++) {
    const m = nn.MLP2(2, 6, 2);
    const Xx = [[0,0],[0,1],[1,0],[1,1]];
    const yx = [0, 1, 1, 0];
    const r = m.train(Xx, yx, { lr: 0.5, iters: 4000 });
    acc = r.accuracy; loss = r.loss;
    if (acc >= 0.999) solved = true;
  }
  check('MLP solves XOR (accuracy = 1.0)', solved, `bestAcc=${acc.toFixed(3)} loss=${loss.toFixed(3)}`);
}

/* 9. Attention — gradient check */
console.log('\n[9] Attention (scaled dot-product)');
{
  const n = 5, d = 4;
  const Q = rndMat(n, d, -1, 1), K = rndMat(n, d, -1, 1), V = rndMat(n, d, -1, 1);
  const sdp = attention.SDP();
  const fwd = (q, k, v) => sdp.forward(q, k, v);
  // scalar loss = sum of out entries; analytic grad via backward(dOut=ones)
  const F = (qf, kf, vf) => {
    const q = M.fromFlat(qf, n, d), k = M.fromFlat(kf, n, d), v = M.fromFlat(vf, n, d);
    const o = fwd(q, k, v).out;
    let s = 0; for (const row of o) for (const val of row) s += val;
    return s;
  };
  const ctx = fwd(Q, K, V);
  const dOut = M.fromFlat(new Array(n * d).fill(1), n, d);
  const bwd = sdp.backward(dOut, { ...ctx, Q, K, V });
  const anaFlat = [].concat(M.toFlat(bwd.dQ), M.toFlat(bwd.dK), M.toFlat(bwd.dV));
  const x0 = [].concat(M.toFlat(Q), M.toFlat(K), M.toFlat(V));
  // numerical gradient over the whole concatenated param vector
  const num = optim.numericalGradient((xf) => {
    const q = M.fromFlat(xf.slice(0, n * d), n, d);
    const k = M.fromFlat(xf.slice(n * d, 2 * n * d), n, d);
    const v = M.fromFlat(xf.slice(2 * n * d), n, d);
    return F(M.toFlat(q), M.toFlat(k), M.toFlat(v));
  }, x0, 1e-6);
  let maxRel = 0;
  for (let i = 0; i < x0.length; i++) {
    const a = anaFlat[i], nm = num[i];
    maxRel = Math.max(maxRel, Math.abs(a - nm) / (Math.abs(a) + Math.abs(nm) + 1e-12));
  }
  check('Attention analytic grad matches numerical', maxRel < 1e-4, `maxRelErr=${maxRel.toExponential(3)}`);
  // softmax over rows property
  check('Attention weights sum to 1 per query', ctx.A.every(row => close(row.reduce((s, v) => s + v, 0), 1, 1e-12)));
}

/* Summary */
console.log(`\n=== RESULT: ${pass} passed, ${fail} failed ===`);
if (fail > 0) { console.log('FAILED: ' + fails.join(', ')); process.exit(1); }
else { console.log('ALL GREEN ✅'); }
