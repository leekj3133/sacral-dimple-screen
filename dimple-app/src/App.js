// src/App.js
import React, { useRef, useState, useEffect } from 'react';
import { loadModel, predictFromImage } from './tfModel';
import './App.css';

function App() {
  const imgRef = useRef(null);
  const [status, setStatus] = useState('Model not loaded');
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleLoad = async () => {
    setStatus('Loading model...');
    await loadModel();
    setStatus('Model ready');
  };

  const onFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    handleFile(file);
    e.target.value = ''; // allow reselecting same file
  };

  const handleFile = (file) => {
    const url = URL.createObjectURL(file);
    setResult(null);
    setStatus('Image selected');
    setPreview(url);
  };

  // Drag & Drop handlers
  const onDragOver = (e) => {
    e.preventDefault();
  };
  const onDragEnter = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const onDragLeave = (e) => {
    e.preventDefault();
    // only clear when leaving the dropzone, not children
    if (e.currentTarget === e.target) setIsDragging(false);
  };
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    handleFile(file);
  };

  useEffect(() => {
    if (!preview) return;
    const img = imgRef.current;
    if (!img) return;

    const run = async () => {
      setStatus('Predicting...');
      try {
        const out = await predictFromImage(img);
        setResult(out);
        setStatus('Done');
      } catch (err) {
        console.error(err);
        setStatus('Error: ' + (err?.message || 'failed'));
      }
    };

    if (img.complete) run();
    else {
      img.onload = () => { run(); img.onload = null; };
      img.onerror = (err) => {
        console.error(err);
        setStatus('Error: image load failed');
        img.onerror = null;
      };
    }

    return () => { if (img) { img.onload = null; img.onerror = null; } };
  }, [preview]);

  return (
    <div className="wrap">
      <header>
        <h1>Sacral Dimple Screen</h1>
        <p>Upload or capture a photo to screen for Normal / Abnormal.</p>
      </header>

      <section className="card">
        <div className="row">
          <button className="primary" onClick={handleLoad}>Load Model</button>
          <p className="status">{status}</p>
        </div>

        {/* Uploader: camera, file picker, drag & drop */}
        <div
          className={`uploader ${isDragging ? 'dragover' : ''}`}
          onDragOver={onDragOver}
          onDragEnter={onDragEnter}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          {/* Hidden inputs */}
          <input
            id="capture-input"
            type="file"
            accept="image/*"
            capture="environment"        // mobile: open camera (rear) by default; desktop ignores
            onChange={onFile}
            style={{ display: 'none' }}
          />
          <input
            id="pick-input"
            type="file"
            accept="image/*"
            onChange={onFile}
            style={{ display: 'none' }}
          />

          {/* Buttons */}
          <div className="btn-row">
            <label htmlFor="capture-input" className="upload-btn">
              Take Photo
            </label>
            <label htmlFor="pick-input" className="upload-btn secondary">
              Select Image
            </label>
          </div>

          <div className="hint">
            or drag &amp; drop an image here
          </div>
        </div>

        {preview && (
          <div className="preview">
            <img ref={imgRef} src={preview} alt="preview" />
          </div>
        )}

        {result && <ResultPanel result={result} />}
      </section>

      <footer>
        <small>
          This tool is for demo/educational purposes only. It is not a medical device
          and does not replace clinical judgment.
        </small>
      </footer>
    </div>
  );
}

function ResultPanel({ result }) {
  const { label, confidence, score } = result;
  const tag = confidenceTag(confidence);     // 'High' | 'Medium' | 'Low'
  const tagText = confidenceBadgeText(tag);  // 'High confidence', etc.
  const guide = guidanceText(label, tag);    // clinician-friendly text

  return (
    <div className={`result ${label === 'Abnormal' ? 'bad' : 'good'}`}>
      <div className="label">{label}</div>
      <span className={`badge conf-${tag.toLowerCase()}`}>{tagText}</span>

      <div className="bar">
        <div className="fill" style={{ width: `${(score * 100).toFixed(0)}%` }} />
      </div>

      <p className="guide">{guide}</p>
    </div>
  );
}

function confidenceTag(c) {
  if (c >= 0.75) return 'High';
  if (c >= 0.50) return 'Medium';
  return 'Low';
}

function confidenceBadgeText(tag) {
  if (tag === 'High') return 'High confidence';
  if (tag === 'Medium') return 'Medium confidence';
  return 'Low confidence';
}

function guidanceText(label, tag) {
  if (label === 'Abnormal') {
    if (tag === 'High')   return 'High likelihood of positive finding. Further evaluation is recommended.';
    if (tag === 'Medium') return 'A positive finding is possible. Consider additional review.';
    return 'Classified as positive, but confidence is low. Re-evaluation with clinical context is advised.';
  } else { // Normal
    if (tag === 'High')   return 'High likelihood of normal. Consider follow-up as appropriate to the clinical context.';
    if (tag === 'Medium') return 'Near the decision threshold. Interpret together with clinical findings.';
    return 'Classified as normal, but confidence is low. Reassessment may be considered if clinically indicated.';
  }
}

export default App;
