import * as tf from '@tensorflow/tfjs';

let model = null;
let modelType = null;
const INPUT_SIZE = 224;

const THRESHOLD_DEFAULT = 0.135; // 0.135
const LOW  = 0.109233;
const HIGH = 0.233867;

// ↓ 필요시 바꿔보는 스위치
// 'normal_abnormal'  : data=[p_norm, p_abn]  → raw = data[1]
// 'abnormal_normal'  : data=[p_abn , p_norm] → raw = data[0]
// const SOFTMAX_ORDER = 'normal_abnormal';   // ← 안 맞으면 'abnormal_normal' 로 바꿔보세요.
 const DEFAULT_ORDER = 'normal_abnormal'; // or 'abnormal_normal'
 const ORDER = (() => {
   if (typeof window === 'undefined') return DEFAULT_ORDER;
   const v = new URLSearchParams(window.location.search).get('order');
   return (v === 'abnormal_normal' || v === 'normal_abnormal') ? v : DEFAULT_ORDER;
 })();
let USE_DIV255 = true;  
let USE_BGR    = false;

const MODEL_URL = '/web_model/model.json'; // 실제 경로 확인

function clamp01(v){ return Math.max(0, Math.min(1, v)); }

// URL 쿼리나 localStorage로 임계값 오버라이드 가능 (현장 튜닝용)
function getThreshold() {
  const m = typeof window !== 'undefined' && window.location.search.match(/[?&]thr=([0-9.]+)/i);
  if (m) return parseFloat(m[1]);
  if (typeof localStorage !== 'undefined') {
    const v = localStorage.getItem('threshold');
    if (v) return parseFloat(v);
  }
  return THRESHOLD_DEFAULT;
}

async function detectModelType(url) {
  const res = await fetch(url);
  const txt = await res.text();
  const j = JSON.parse(txt);
  const fmt = (j.format || '').toLowerCase();
  if (fmt.includes('graph'))  return 'graph';
  if (fmt.includes('layers')) return 'layers';
  if (j.modelTopology)        return 'layers';
  throw new Error('Unknown model.json format');
}

export async function loadModel() {
  if (model) return model;
  modelType = await detectModelType(MODEL_URL);
  console.log('[tfModel] detected type =', modelType);

  if (modelType === 'graph') {
    model = await tf.loadGraphModel(MODEL_URL);
    console.log('[tfModel] graph inputs:', model.inputs?.map(t => t.name));
    console.log('[tfModel] graph outputs:', model.outputs?.map(t => t.name));
  } else {
    model = await tf.loadLayersModel(MODEL_URL);
    console.log('[tfModel] layers inputs:', model.inputs?.map(t => t.name));
    console.log('[tfModel] layers outputs:', model.outputs?.map(t => t.name));
  }
  console.log('[tfModel] thresholds:', { THRESHOLD_DEFAULT, LOW, HIGH });
  return model;
}

function preprocess(img, div255 = USE_DIV255, toBGR = USE_BGR) {
  let t = tf.browser.fromPixels(img).toFloat();

  // OpenCV INTER_LINEAR 과 유사하게 bilinear 사용
  t = tf.image.resizeBilinear(
    t,
    [INPUT_SIZE, INPUT_SIZE],
    /*alignCorners=*/false // cv2 기본과 더 유사
  );

  if (toBGR) {
    const [r, g, b] = tf.split(t, 3, -1);
    t = tf.concat([b, g, r], -1);
  }

  if (div255) t = t.div(255);

  return t.expandDims(); // (1, H, W, 3)
}

export async function predictFromImage(img) {
  await loadModel();

  const x = preprocess(img);
  const xMin = (await x.min().data())[0];
  const xMax = (await x.max().data())[0];
  console.log('[tfModel] x range:', xMin, '→', xMax, 'div255=', USE_DIV255, 'BGR=', USE_BGR);

  let y;
  if (modelType === 'graph') {
    try {
      y = model.execute(x);
    } catch {
      const sig = model.executor?.graph?.signature;
      if (!sig) throw new Error('No graph signature');
      const inName  = Object.keys(sig.inputs)[0];
      const outName = Object.keys(sig.outputs)[0];
      console.log('[tfModel] execute with names:', inName, '->', outName);
      y = model.execute({ [inName]: x }, outName);
    }
  } else {
    y = model.predict(x);
  }

  const outs = Array.isArray(y) ? y : [y];
  const t0 = outs[0];
  const data = await t0.data();
  console.log('[tfModel] out shape=', t0.shape, 'len=', data.length, 'first10=', Array.from(data).slice(0,10));


let raw;
if (data.length === 1) {
  // 1-channel (sigmoid or logit)
  // logit일 수도 있으니 sigmoid 변환값도 로그로 확인
  const asSigmoid = 1 / (1 + Math.exp(-data[0]));
  console.log('[tfModel] len=1 first=', data[0], 'sigmoid(first)=', asSigmoid);
  // 보통은 이미 sigmoid일 가능성이 높아 data[0] 사용
  raw = data[0];
} else if (data.length === 2) {
  const p0 = data[0], p1 = data[1];
  const sum = p0 + p1;
  console.log('[tfModel] len=2 p0=', p0, 'p1=', p1, 'sum≈', sum);

  // softmax라면 sum이 1에 가깝다. 아니라면 logits 가능성 → sigmoid로 바꿔보기
  if (Math.abs(sum - 1) < 1e-3) {
    // softmax로 가정
    // raw = (SOFTMAX_ORDER === 'normal_abnormal') ? p1 : p0;
    raw = (ORDER === 'normal_abnormal') ? p1 : p0;
  } else {
    // logits로 가정: 각 채널에 sigmoid 적용 후 다시 선택
    const s0 = 1 / (1 + Math.exp(-p0));
    const s1 = 1 / (1 + Math.exp(-p1));
    console.log('[tfModel] treated as logits s0=', s0, 's1=', s1);
    // raw = (SOFTMAX_ORDER === 'normal_abnormal') ? s1 : s0;
    raw = (ORDER === 'normal_abnormal') ? s1 : s0;
  }
} else {
  console.warn('[tfModel] Unexpected output length:', data.length, '→ using data[0]');
  raw = data[0];
}

console.log('[tfModel] raw after selection =', raw);
  const thr = getThreshold();
  const label = raw > thr ? 'Abnormal' : 'Normal';
  const score = clamp01((raw - LOW) / (HIGH - LOW));
  const confidence = label === 'Abnormal' ? score : (1 - score);

  console.log('[tfModel] raw=', raw.toFixed(6), 'thr=', thr, 'label=', label,
              'score=', score.toFixed(3), 'conf=', confidence.toFixed(3));

  x.dispose();
  outs.forEach(t => t.dispose());

  return { label, confidence, score, raw };
}

export function setPreprocess({ div255, bgr }) {
  if (typeof div255 === 'boolean') USE_DIV255 = div255;
  if (typeof bgr === 'boolean')    USE_BGR    = bgr;
  console.log('[tfModel] preprocess switches:', { USE_DIV255, USE_BGR });
}