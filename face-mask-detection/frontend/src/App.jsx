import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { io } from "socket.io-client";

const API_URL = "http://localhost:5000/predict";
const SOCKET_URL = "http://localhost:5000";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [boxes, setBoxes] = useState([]); // [{x,y,w,h,label,confidence}]
  const [imageSize, setImageSize] = useState([360, 288]); // width, height of last sent image

  const [useWebcam, setUseWebcam] = useState(false);
  const [autoCapture, setAutoCapture] = useState(false);
  const [useSocket, setUseSocket] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("idle"); // idle | checking | ready | denied | no-device | unsupported | error

  const webcamRef = useRef(null);
  const autoIntervalRef = useRef(null);
  const socketRef = useRef(null);

  useEffect(() => {
    if (useSocket && !socketRef.current) {
      socketRef.current = io(SOCKET_URL);
      socketRef.current.on("connect", () =>
        console.log("Socket connected", socketRef.current.id),
      );
      socketRef.current.on("prediction", (data) => {
        setLoading(false);
        if (data.error) setError(data.error);
        else {
          setResult(
            data.detections && data.detections[0] ? data.detections[0] : null,
          );
          setBoxes(data.detections || []);
          if (data.image_size) setImageSize(data.image_size);
        }
      });
      socketRef.current.on("connect_error", (err) =>
        setError("Socket connect error: " + err.message),
      );
    }
    return () => {
      if (socketRef.current && !useSocket) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, [useSocket]);

  useEffect(() => {
    if (autoCapture && useWebcam) {
      autoIntervalRef.current = setInterval(() => captureAndSend(), 1000);
    } else {
      clearInterval(autoIntervalRef.current);
    }
    return () => clearInterval(autoIntervalRef.current);
  }, [autoCapture, useWebcam, useSocket]);

  // Check camera availability & permissions when user toggles webcam
  useEffect(() => {
    let cancelled = false;
    async function checkCamera() {
      setCameraStatus("checking");
      setError(null);

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraStatus("unsupported");
        setError("Camera not supported in this browser.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        // stop immediately — we only check permission/device
        stream.getTracks().forEach((t) => t.stop());
        if (!cancelled) {
          setCameraStatus("ready");
          setError(null);
        }
      } catch (err) {
        if (err.name === "NotAllowedError" || err.name === "SecurityError") {
          setCameraStatus("denied");
          setError("Camera access denied. Allow camera in your browser.");
        } else if (
          err.name === "NotFoundError" ||
          err.name === "OverconstrainedError"
        ) {
          setCameraStatus("no-device");
          setError("No camera device found.");
        } else {
          setCameraStatus("error");
          setError("Camera error: " + err.message);
        }
      }
    }

    if (useWebcam) checkCamera();
    else {
      setCameraStatus("idle");
      setError(null);
    }

    return () => {
      cancelled = true;
    };
  }, [useWebcam]);

  const submitFile = async (e) => {
    e.preventDefault();
    if (!file) return;
    await predictFile(file);
  };

  const predictFile = async (fileBlob) => {
    if (useSocket && socketRef.current?.connected) {
      setLoading(true);
      setError(null);
      setResult(null);

      const reader = new FileReader();
      reader.onload = () => {
        socketRef.current.emit("frame", { image: reader.result });
      };
      reader.readAsDataURL(fileBlob);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const fd = new FormData();
    fd.append("image", fileBlob, "upload.jpg");

    try {
      const res = await fetch(API_URL, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      // server now returns detections array + image_size
      setResult(
        json.detections && json.detections[0] ? json.detections[0] : null,
      );
      setBoxes(json.detections || []);
      if (json.image_size) setImageSize(json.image_size);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const captureAndSend = async () => {
    if (!webcamRef.current) return;
    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) return;

    if (useSocket && socketRef.current?.connected) {
      setLoading(true);
      setError(null);
      setResult(null);
      socketRef.current.emit("frame", { image: screenshot });
      return;
    }

    // fallback: send via REST
    try {
      const res = await fetch(screenshot);
      const blob = await res.blob();
      await predictFile(blob);
    } catch (err) {
      setError("Failed to capture/send frame: " + err.message);
    } finally {
      // draw boxes cleared when no detections
      // boxes will be populated by predictFile result
    }
  };

  return (
    <div className="container">
      <h1>Face Mask Detector</h1>

      <div className="card" style={{ flexDirection: "column", gap: 12 }}>
        <form
          onSubmit={submitFile}
          style={{ display: "flex", gap: 12, width: "100%" }}
        >
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button type="submit" disabled={!file || loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        <div
          style={{
            display: "flex",
            gap: 12,
            alignItems: "center",
            marginTop: 8,
          }}
        >
          <button
            onClick={() => {
              setUseWebcam((v) => {
                const nv = !v;
                if (!nv) setAutoCapture(false);
                return nv;
              });
            }}
          >
            {useWebcam ? "Stop Webcam" : "Start Webcam"}
          </button>

          <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={autoCapture}
              onChange={(e) => setAutoCapture(e.target.checked)}
              disabled={!useWebcam}
            />
            Auto-capture (1s)
          </label>

          <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={useSocket}
              onChange={(e) => setUseSocket(e.target.checked)}
            />
            Use WebSocket
          </label>

          {useWebcam && (
            <button
              onClick={captureAndSend}
              disabled={loading || cameraStatus !== "ready"}
            >
              Capture & Predict
            </button>
          )}

          <div style={{ marginLeft: 8 }}>
            {cameraStatus === "checking" && <small>Checking camera…</small>}
            {cameraStatus === "ready" && (
              <small style={{ color: "#5eead4" }}>Camera ready</small>
            )}
            {cameraStatus === "denied" && (
              <small style={{ color: "#f87171" }}>
                Permission denied — allow camera
              </small>
            )}
            {cameraStatus === "no-device" && (
              <small style={{ color: "#f97316" }}>No camera found</small>
            )}
            {cameraStatus === "unsupported" && (
              <small style={{ color: "#f97316" }}>
                Browser does not support camera
              </small>
            )}
            {cameraStatus === "error" && (
              <small style={{ color: "#f97316" }}>Camera error</small>
            )}
          </div>
        </div>
      </div>

      {useWebcam && (
        <div
          className="card"
          style={{
            marginTop: 12,
            justifyContent: "center",
            position: "relative",
          }}
        >
          <Webcam
            audio={false}
            ref={webcamRef}
            onUserMedia={() => {
              setCameraStatus("ready");
              setError(null);
              setAutoCapture(true); // start auto-detection when webcam is ready
            }}
            onUserMediaError={(e) => {
              setCameraStatus("error");
              setError("Webcam error: " + (e?.message || e));
            }}
            screenshotFormat="image/jpeg"
            width={360}
            videoConstraints={{ width: 360, height: 288, facingMode: "user" }}
          />

          {/* overlay: draw boxes using percentages relative to returned image_size */}
          <div
            style={{
              position: "absolute",
              left: "50%",
              transform: "translateX(-50%)",
              top: 8,
              width: 360,
              height: 288,
            }}
          >
            {boxes.map((b, i) => {
              const [imgW, imgH] = imageSize || [360, 288];
              const leftPct = (b.box[0] / imgW) * 100;
              const topPct = (b.box[1] / imgH) * 100;
              const wPct = (b.box[2] / imgW) * 100;
              const hPct = (b.box[3] / imgH) * 100;
              const style = {
                position: "absolute",
                left: leftPct + "%",
                top: topPct + "%",
                width: wPct + "%",
                height: hPct + "%",
                border:
                  "2px solid " + (b.label === "Mask" ? "#34d399" : "#fb7185"),
                boxSizing: "border-box",
                pointerEvents: "none",
              };
              return (
                <div key={i} style={style}>
                  <div
                    style={{
                      background: "rgba(0,0,0,0.6)",
                      color: "#fff",
                      fontSize: 12,
                      padding: "2px 6px",
                      borderRadius: 4,
                      position: "absolute",
                      left: 0,
                      top: "-22px",
                    }}
                  >
                    {b.label} ({(b.confidence * 100).toFixed(0)}%)
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result card">
          <h2>Result</h2>
          <p>
            <strong>Prediction:</strong> {result.label}
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
          </p>
          <pre className="debug">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}

      <footer>
        <small></small>
      </footer>
    </div>
  );
}
