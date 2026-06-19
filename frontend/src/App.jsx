import React, { useState, useRef, useEffect } from 'react';

const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [status, setStatus] = useState('idle'); // 'idle' | 'recording' | 'processing'
  const [transcription, setTranscription] = useState('');
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [copied, setCopied] = useState(false);
  
  // v3.1 New State variables
  const [keywords, setKeywords] = useState('');
  const [metadata, setMetadata] = useState(null);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [showSettings, setShowSettings] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);
  
  // Refs for Web Audio API to prevent leaks
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameIdRef = useRef(null);
  const mediaStreamRef = useRef(null);

  // Clean up audio references on unmount
  useEffect(() => {
    return () => {
      if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
      if (audioContextRef.current) {
        try {
          audioContextRef.current.close();
        } catch (_) {}
      }
    };
  }, []);

  // Microphone volume analyser
  const startVolumeAnalysis = (stream) => {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContextClass();
      audioContextRef.current = audioCtx;
      
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      
      source.connect(analyser);
      
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      const checkVolume = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteFrequencyData(dataArray);
        
        let total = 0;
        for (let i = 0; i < bufferLength; i++) {
          total += dataArray[i];
        }
        const average = total / bufferLength;
        // Map typical voice amplitude average (~0-120) to 0-100 percentage
        const scale = Math.min(100, Math.round((average / 120) * 100));
        setVolumeLevel(scale);
        
        animationFrameIdRef.current = requestAnimationFrame(checkVolume);
      };
      
      animationFrameIdRef.current = requestAnimationFrame(checkVolume);
    } catch (err) {
      console.warn("Web Audio API failed to initialize:", err);
    }
  };

  const stopVolumeAnalysis = () => {
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (_) {}
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setVolumeLevel(0);
  };

  // Microphone recording
  const startRecording = async () => {
    setError(null);
    setTranscription('');
    setMetadata(null);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      
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
        stopVolumeAnalysis();
        const mimeType = recorder.mimeType || 'audio/webm';
        const extension = mimeType.includes('mp4') ? 'mp4' : 'webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        await uploadAudio(audioBlob, `recording.${extension}`);
        
        stream.getTracks().forEach(track => track.stop());
      };

      // Start volume analysis
      startVolumeAnalysis(stream);

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
      stopVolumeAnalysis();
      setStatus('processing');
    }
  };

  // Upload file or recorded blob to API
  const uploadAudio = async (blob, filename) => {
    setStatus('processing');
    setError(null);

    const formData = new FormData();
    formData.append('file', blob, filename);
    
    // Add optional keywords spelling hints
    if (keywords.trim()) {
      formData.append('keywords', keywords.trim());
    }

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
      setMetadata({
        num_segments: result.num_segments,
        confidence: result.confidence,
        word_count: result.word_count,
        char_count: result.char_count
      });
      setStatus('idle');
    } catch (err) {
      console.error("Transcription upload error:", err);
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
        setMetadata(null);
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
      setMetadata(null);
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

  // Keyboard shortcuts listener
  useEffect(() => {
    const handleKeyDown = (e) => {
      const activeElement = document.activeElement;
      const isTyping = activeElement && (
        activeElement.tagName === 'INPUT' || 
        activeElement.tagName === 'TEXTAREA' || 
        activeElement.isContentEditable
      );

      // Space: Start/Stop recording (only when not typing)
      if (e.code === 'Space' && !isTyping) {
        e.preventDefault();
        if (status === 'idle') {
          startRecording();
        } else if (status === 'recording') {
          stopRecording();
        }
      }

      // Esc: Reset ASR state / clear file
      if (e.code === 'Escape') {
        setSelectedFile(null);
        setError(null);
        setTranscription('');
        setMetadata(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }

      // Ctrl + C / Cmd + C: Copy transcription text (only if no custom text selection exists)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') {
        if (transcription && !isTyping) {
          const selection = window.getSelection().toString();
          if (!selection) {
            e.preventDefault();
            copyToClipboard();
          }
        }
      }

      // Ctrl + D / Cmd + D: Download transcription (.txt)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') {
        if (transcription) {
          e.preventDefault(); // Stop default browser bookmark popup
          downloadTextFile();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [status, transcription, selectedFile, keywords]);

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
              
              {/* Dynamic volume level bar */}
              <div className="volume-meter-container">
                <div 
                  className="volume-meter-bar" 
                  style={{ 
                    width: `${volumeLevel}%`,
                    opacity: 0.3 + (volumeLevel / 100) * 0.7
                  }}
                ></div>
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
            {/* Collapsible Spelling hints setting */}
            <div className="settings-container">
              <button 
                className="settings-toggle-btn"
                onClick={() => setShowSettings(!showSettings)}
              >
                <svg viewBox="0 0 24 24" className={showSettings ? 'rotated' : ''}>
                  <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
                </svg>
                Spelling hints
              </button>
              
              <div className={`settings-drawer ${showSettings ? 'open' : ''}`}>
                <div className="settings-content">
                  <input
                    id="spelling-hints-input"
                    type="text"
                    placeholder="Names, technical terms, key phrases (e.g. ស្វាគមន៍, បំប្លែង)"
                    value={keywords}
                    onChange={(e) => setKeywords(e.target.value)}
                  />
                </div>
              </div>
            </div>

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

            {/* ASR Quality Metadata Row */}
            {metadata && (
              <div className="metadata-row">
                Khmer • {metadata.confidence}% confidence • {metadata.num_segments} segments • {metadata.word_count} words • {metadata.char_count} characters
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
