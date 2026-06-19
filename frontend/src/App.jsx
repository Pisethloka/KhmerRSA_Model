import React, { useState, useRef } from 'react';

const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [status, setStatus] = useState('idle'); // 'idle' | 'recording' | 'processing'
  const [transcription, setTranscription] = useState('');
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [copied, setCopied] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  // Microphone recording
  const startRecording = async () => {
    setError(null);
    setTranscription('');
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      let options = { mimeType: 'audio/webm' };
      if (!MediaRecorder.isTypeSupported('audio/webm')) {
        options = { mimeType: 'audio/mp4' };
        console.warn("audio/webm not supported. Falling back to audio/mp4.");
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
        
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setStatus('recording');
    } catch (err) {
      console.error("Microphone access error:", err);
      setError("Unable to access microphone. Please check browser permissions and connections.");
      setStatus('idle');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && status === 'recording') {
      mediaRecorderRef.current.stop();
      setStatus('processing');
    }
  };

  // Upload file or recording to backend
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
      setStatus('idle');
    } catch (err) {
      console.error("Upload error:", err);
      setError(err.message || "Connection failed. Please ensure the backend server is running.");
      setStatus('idle');
    }
  };

  // Drag and drop
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
      } else {
        setError("Unsupported file type. Please select an audio file (WAV, MP3, M4A, etc.).");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
      setTranscription('');
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
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadTextFile = () => {
    const element = document.createElement("a");
    const file = new Blob([transcription], {type: 'text/plain;charset=utf-8'});
    element.href = URL.createObjectURL(file);
    element.download = "khmer_transcription.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="container">
      {/* Title Header */}
      <div className="header-section">
        <h1>Khmer Speech to Text</h1>
        <p className="subtitle">Convert spoken Khmer into text</p>
      </div>

      <div className="asr-card">
        {/* Recording section */}
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
              <div className="status-text">Click microphone to record</div>
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
              <div className="status-text recording">Recording... Click to stop</div>
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
              <div className="status-text processing">Processing...</div>
            </>
          )}
        </div>

        {status === 'idle' && (
          <>
            {/* Clean Divider */}
            <div className="divider">
              <span className="divider-text">or</span>
            </div>

            {/* Notion-style Dropzone */}
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
                  <p>Upload audio file</p>
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
                  Transcribe
                </button>
              )}
            </div>
          </>
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
              <span className="result-title">Transcription</span>
            </div>
            
            <textarea 
              className="result-box" 
              value={transcription}
              onChange={(e) => setTranscription(e.target.value)}
              placeholder="Transcription will appear here..."
              id="transcription-output-box"
            />
            
            <div className="actions-row">
              <button className="action-btn" onClick={copyToClipboard}>
                {copied ? (
                  <>
                    <svg viewBox="0 0 24 24">
                      <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                    </svg>
                    Copied
                  </>
                ) : (
                  <>
                    <svg viewBox="0 0 24 24">
                      <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
                    </svg>
                    Copy Text
                  </>
                )}
              </button>
              
              <button className="action-btn" onClick={downloadTextFile}>
                <svg viewBox="0 0 24 24">
                  <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/>
                </svg>
                Download TXT
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
