import React, { useState, useRef } from 'react';

const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [status, setStatus] = useState('idle'); // 'idle' | 'recording' | 'processing'
  const [transcription, setTranscription] = useState('');
  const [durationMs, setDurationMs] = useState(null);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  // Microphone recording functions
  const startRecording = async () => {
    setError(null);
    setTranscription('');
    setDurationMs(null);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Determine best format supported, defaulting to audio/webm per spec
      let options = { mimeType: 'audio/webm' };
      if (!MediaRecorder.isTypeSupported('audio/webm')) {
        // Fallback for Safari/unsupported browsers
        options = { mimeType: 'audio/mp4' };
        console.warn("audio/webm not supported on this browser. Falling back to audio/mp4.");
      }

      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const mimeType = recorder.mimeType || 'audio/webm';
        const extension = mimeType.includes('mp4') ? 'mp4' : 'webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        await uploadAudio(audioBlob, `recording.${extension}`);
        
        // Stop all tracks to turn off mic light
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setStatus('recording');
    } catch (err) {
      console.error("Microphone access error:", err);
      setError("Unable to access microphone. Please check browser permissions and ensure a microphone is connected.");
      setStatus('idle');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && status === 'recording') {
      mediaRecorderRef.current.stop();
      setStatus('processing');
    }
  };

  // Upload file or recorded blob to API
  const uploadAudio = async (blob, filename) => {
    setStatus('processing');
    setError(null);

    const formData = new FormData();
    formData.append('file', blob, filename);

    try {
      const response = await fetch(`${BACKEND_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `Server error (${response.status})`;
        try {
          const parsed = JSON.parse(errorText);
          errorMessage = parsed.detail || errorMessage;
        } catch (_) {}
        throw new Error(errorMessage);
      }

      const result = await response.json();
      setTranscription(result.text);
      setDurationMs(result.duration_ms);
      setStatus('idle');
    } catch (err) {
      console.error("Transcription upload error:", err);
      setError(err.message || "Connection failed. Please ensure the backend server is running and try again.");
      setStatus('idle');
    }
  };

  // Drag and drop event handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('audio/') || file.name.match(/\.(mp3|wav|m4a|mp4|webm|ogg)$/i)) {
        setSelectedFile(file);
        setError(null);
        setTranscription('');
        setDurationMs(null);
      } else {
        setError("Unsupported file type. Please select an audio file (WAV, MP3, M4A, WebM, etc.).");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
      setTranscription('');
      setDurationMs(null);
    }
  };

  const clearSelectedFile = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleTranscribeFile = async () => {
    if (selectedFile) {
      await uploadAudio(selectedFile, selectedFile.name);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(transcription);
    alert("Copied transcription to clipboard!");
  };

  const formatDuration = (ms) => {
    if (ms === null || ms === undefined) return '';
    const seconds = (ms / 1000).toFixed(1);
    return `${seconds}s`;
  };

  return (
    <div className="container">
      <div className="asr-card">
        <div>
          <h1>ប្រព័ន្ធបំប្លែងសំឡេងជាអក្សរខ្មែរ</h1>
          <p className="subtitle">Khmer Automatic Speech Recognition (ASR)</p>
        </div>

        {/* Status display */}
        <div className="recorder-section">
          {status === 'idle' && (
            <>
              <button 
                id="record-button-idle"
                className="record-btn" 
                onClick={startRecording}
                title="Start Recording"
              >
                <svg viewBox="0 0 24 24">
                  <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                  <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                </svg>
              </button>
              <div className="status-text">ចុចដើម្បីកត់ត្រាសំឡេង (Click to Record)</div>
            </>
          )}

          {status === 'recording' && (
            <>
              <button 
                id="record-button-recording"
                className="record-btn recording" 
                onClick={stopRecording}
                title="Stop Recording"
              >
                <svg viewBox="0 0 24 24">
                  <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
                </svg>
              </button>
              <div className="status-text recording">កំពុងថត... (Recording...)</div>
              <div className="wave-container">
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
              </div>
            </>
          )}

          {status === 'processing' && (
            <>
              <div className="spinner"></div>
              <div className="status-text processing">កំពុងបកប្រែសំឡេង... (Processing...)</div>
            </>
          )}
        </div>

        {/* Fallback File Uploader */}
        {status === 'idle' && (
          <div 
            className={`dropzone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              id="file-upload-input"
              ref={fileInputRef}
              type="file" 
              className="file-input" 
              accept="audio/*,video/*"
              onChange={handleFileChange}
            />
            
            {selectedFile ? (
              <div className="selected-file-badge">
                <span>{selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)</span>
                <button onClick={clearSelectedFile} title="Clear file">&times;</button>
              </div>
            ) : (
              <>
                <svg viewBox="0 0 24 24">
                  <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
                </svg>
                <p>អូស និងទម្លាក់ឯកសារសំឡេង ឬចុចដើម្បីជ្រើសរើស</p>
                <p style={{ fontSize: '0.8rem', opacity: 0.75 }}>Drag & Drop audio file here or click to browse</p>
              </>
            )}

            {selectedFile && (
              <button 
                id="transcribe-file-button"
                className="transcribe-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  handleTranscribeFile();
                }}
              >
                បកប្រែឯកសារ (Transcribe)
              </button>
            )}
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div className="error-alert">
            <svg viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
            </svg>
            <div>{error}</div>
          </div>
        )}

        {/* Result Area */}
        {transcription && (
          <div className="result-section">
            <div className="result-header">
              <span className="result-title">លទ្ធផលបកប្រែ (Transcription Result)</span>
              {durationMs && (
                <span className="meta-info">Duration: {formatDuration(durationMs)}</span>
              )}
            </div>
            <div className="result-box" id="transcription-output-box">
              {transcription}
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button className="copy-btn" onClick={copyToClipboard}>
                <svg viewBox="0 0 24 24" style={{ width: 14, height: 14, fill: 'currentColor' }}>
                  <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
                </svg>
                ចម្លងអត្ថបទ (Copy Text)
              </button>
            </div>
          </div>
        )}
      </div>
      <div className="footer-info">
        Khmer ASR Powered by Whisper Large v3
      </div>
    </div>
  );
}

export default App;
