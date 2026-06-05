import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Camera, Upload, AlertTriangle, FileText } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const THREAT_CLASSES = ['Drone', 'AirPlane', 'Helicopter', 'UAV'];
const WATCH_CLASSES = ['Bird'];
const SAFE_CLASSES = ['Person'];

function getThreatColor(className) {
  if (THREAT_CLASSES.some(t => className.toLowerCase().includes(t.toLowerCase()))) return '#ff0055';
  if (WATCH_CLASSES.some(t => className.toLowerCase().includes(t.toLowerCase()))) return '#ffaa00';
  if (SAFE_CLASSES.some(t => className.toLowerCase().includes(t.toLowerCase()))) return '#00cc88';
  return '#00ffc8';
}

export default function CameraDetection() {
    const [mode, setMode] = useState('upload');
    const [mediaUrl, setMediaUrl] = useState(null);
    const [mediaType, setMediaType] = useState(null);
    const [detections, setDetections] = useState([]);
    const [model, setModel] = useState(null);
    const [loading, setLoading] = useState(true);
    const [isStreaming, setIsStreaming] = useState(false);

    const [annotationData, setAnnotationData] = useState(null);
    const [annotationFps] = useState(30);
    const [annotationFileName, setAnnotationFileName] = useState(null);

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const imageRef = useRef(null);
    const streamRef = useRef(null);

    useEffect(() => {
        cocoSsd.load().then(m => {
            setModel(m);
            setLoading(false);
        });
    }, []);

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const url = URL.createObjectURL(file);
        setMediaUrl(url);
        setMediaType(file.type.startsWith('video') ? 'video' : 'image');
        setDetections([]);
        setAnnotationData(null);
        setAnnotationFileName(null);
    };

    const handleAnnotationUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setAnnotationFileName(file.name);
        const reader = new FileReader();
        reader.onload = (evt) => {
            const text = evt.target.result;
            const lines = text.trim().split('\n');
            const boxes = lines.map(line => {
                const parts = line.split(',').map(Number);
                if (parts.length === 4 && !isNaN(parts[0])) {
                    return { x: parts[0], y: parts[1], w: parts[2], h: parts[3] };
                }
                return null;
            }).filter(b => b !== null && !(b.x === 0 && b.y === 0 && b.w === 0 && b.h === 0));
            setAnnotationData(boxes);
        };
        reader.readAsText(file);
    };

    const detectInImage = async () => {
        if (!model || !imageRef.current) return;
        const predictions = await model.detect(imageRef.current);
        setDetections(predictions);
        drawDetections(predictions, imageRef.current);
    };

    let animFrameId = null;

    const detectInVideo = async () => {
        if (!model || !videoRef.current) return;
        const detect = async () => {
            if (videoRef.current && !videoRef.current.paused) {
                const predictions = await model.detect(videoRef.current);
                setDetections(predictions);
                const source = videoRef.current;
                const canvas = canvasRef.current;
                if (canvas) {
                    canvas.width = source.videoWidth || source.naturalWidth;
                    canvas.height = source.videoHeight || source.naturalHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    drawDetectionsOnCtx(ctx, predictions, source);
                    drawAnnotationOnCtx(ctx, source);
                }
                animFrameId = requestAnimationFrame(detect);
            }
        };
        detect();
    };

    const drawDetectionsOnCtx = (ctx, predictions, source) => {
        predictions.forEach(p => {
            const [x, y, w, h] = p.bbox;
            const color = getThreatColor(p.class);
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = color;
            const label = `${p.class} ${Math.round(p.score * 100)}%`;
            ctx.fillRect(x, y - 30, ctx.measureText(label).width + 16, 30);
            ctx.fillStyle = '#000';
            ctx.font = '16px Inter';
            ctx.fillText(label, x + 8, y - 8);
        });
    };

    const drawAnnotationOnCtx = (ctx, source) => {
        if (!annotationData || !source) return;
        const fps = annotationFps;
        const frameIndex = Math.floor(source.currentTime * fps);
        if (frameIndex < 0 || frameIndex >= annotationData.length) return;
        const box = annotationData[frameIndex];
        if (!box || box.w === 0) return;
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.setLineDash([]);
        ctx.fillStyle = '#00ff00';
        const label = 'GT';
        ctx.fillRect(box.x, box.y - 26, ctx.measureText(label).width + 16, 26);
        ctx.fillStyle = '#000';
        ctx.font = '14px Inter';
        ctx.fillText(label, box.x + 8, box.y - 8);
        ctx.font = '12px Inter';
        ctx.fillStyle = '#00ff00';
        const coord = `[${box.x},${box.y} ${box.w}x${box.h}]`;
        ctx.fillText(coord, box.x + 8, box.y - 24);
    };

    const drawDetections = (predictions, source) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = source.videoWidth || source.naturalWidth;
        canvas.height = source.videoHeight || source.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawDetectionsOnCtx(ctx, predictions, source);
        drawAnnotationOnCtx(ctx, source);
    };

    const startWebcam = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
                setIsStreaming(true);
                videoRef.current.onloadedmetadata = () => detectLiveVideo();
            }
        } catch (err) {
            console.error('Webcam error:', err);
        }
    };

    const stopWebcam = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            setIsStreaming(false);
        }
        if (animFrameId) cancelAnimationFrame(animFrameId);
    };

    const detectLiveVideo = async () => {
        if (!model || !videoRef.current || !isStreaming) return;
        const predictions = await model.detect(videoRef.current);
        setDetections(predictions);
        drawDetections(predictions, videoRef.current);
        animFrameId = requestAnimationFrame(detectLiveVideo);
    };

    const classCounts = detections.reduce((acc, d) => {
        const name = d.class || d.class_name || 'unknown';
        acc[name] = (acc[name] || 0) + 1;
        return acc;
    }, {});

    const threatCount = detections.filter(d =>
        THREAT_CLASSES.some(t => (d.class || d.class_name || '').toLowerCase().includes(t.toLowerCase()))
    ).length;

    const watchCount = detections.filter(d =>
        WATCH_CLASSES.some(t => (d.class || d.class_name || '').toLowerCase().includes(t.toLowerCase()))
    ).length;

    const total = detections.length;

    return (
        <div className="space-y-6">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass rounded-2xl p-6"
            >
                <h2 className="text-3xl font-bold gradient-text mb-2">AI Vision Detection</h2>
                <p className="text-white/60 text-sm mb-6">
                    Real-time object detection powered by TensorFlow.js
                </p>

                <div className="flex gap-2 mb-6 flex-wrap">
                    {[
                        { id: 'upload', label: 'Upload Media', icon: Upload },
                        { id: 'webcam', label: 'Live Webcam', icon: Camera },
                    ].map(opt => (
                        <button
                            key={opt.id}
                            onClick={() => setMode(opt.id)}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                ${mode === opt.id
                                ? 'bg-horus-cyan text-black'
                                : 'border border-horus-cyan/30 text-horus-cyan'}`}
                        >
                            <opt.icon className="w-4 h-4" />
                            {opt.label}
                        </button>
                    ))}
                </div>

                {/* Upload Mode */}
                {!loading && mode === 'upload' && (
                    <div>
                        {!mediaUrl ? (
                            <div className="space-y-4">
                                <label className="block">
                                    <div className="border-2 border-dashed border-horus-cyan/30 rounded-xl p-12 text-center cursor-pointer hover:border-horus-cyan/60 transition-all">
                                        <Upload className="w-12 h-12 text-horus-cyan/50 mx-auto mb-4" />
                                        <p className="text-white/80 mb-2">Drop an image or video here</p>
                                        <p className="text-xs text-white/40">Supports JPG, PNG, MP4, WebM</p>
                                        <input type="file" accept="image/*,video/*" onChange={handleFileUpload} className="hidden" />
                                    </div>
                                </label>
                                <label className="flex items-center gap-3 p-4 rounded-xl border border-dashed border-white/20 cursor-pointer hover:border-horus-cyan/40 transition-all">
                                    <FileText className="w-6 h-6 text-horus-cyan/60" />
                                    <div>
                                        <p className="text-sm text-white/80">Load annotation.txt</p>
                                        <p className="text-xs text-white/40">Ground truth boxes overlay (one box per frame)</p>
                                    </div>
                                    <input type="file" accept=".txt" onChange={handleAnnotationUpload} className="hidden" />
                                </label>
                                {annotationFileName && (
                                    <p className="text-xs text-horus-cyan flex items-center gap-1">
                                        <FileText className="w-3 h-3" />
                                        Loaded: {annotationFileName} ({annotationData ? annotationData.length : 0} frames)
                                    </p>
                                )}
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="relative rounded-xl overflow-hidden bg-black">
                                    {mediaType === 'image' ? (
                                        <img ref={imageRef} src={mediaUrl} onLoad={detectInImage} className="w-full" alt="Detection target" />
                                    ) : (
                                        <video ref={videoRef} src={mediaUrl} controls onPlay={detectInVideo} className="w-full" />
                                    )}
                                    <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
                                </div>
                                <button onClick={() => { setMediaUrl(null); setDetections([]); setAnnotationData(null); setAnnotationFileName(null); }}
                                    className="px-4 py-2 border border-horus-cyan/50 text-horus-cyan rounded-lg">
                                    Upload another
                                </button>
                            </div>
                        )}
                    </div>
                )}

                {/* Webcam Mode */}
                {!loading && mode === 'webcam' && (
                    <div>
                        <div className="relative rounded-xl overflow-hidden bg-black mb-4">
                            <video ref={videoRef} autoPlay muted playsInline className="w-full" />
                            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
                            {!isStreaming && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                                    <button onClick={startWebcam}
                                        className="px-6 py-3 bg-horus-cyan text-black rounded-lg font-medium">
                                        Start Webcam
                                    </button>
                                </div>
                            )}
                        </div>
                        {isStreaming && (
                            <button onClick={stopWebcam} className="px-4 py-2 bg-horus-red text-white rounded-lg">
                                Stop Webcam
                            </button>
                        )}
                    </div>
                )}

                {loading && (
                    <div className="flex items-center justify-center py-12">
                        <div className="w-8 h-8 border-2 border-horus-cyan border-t-transparent rounded-full animate-spin" />
                        <span className="ml-3 text-horus-cyan">Loading AI model...</span>
                    </div>
                )}
            </motion.div>

            {detections.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass rounded-xl p-6"
                >
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-horus-amber" />
                        Detected Objects ({detections.length})
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {detections.map((d, i) => {
                            const name = d.class || d.class_name || 'unknown';
                            const conf = d.score || d.confidence || 0;
                            const color = getThreatColor(name);
                            return (
                                <div key={i} className="p-3 rounded-lg border" style={{ borderColor: color + '80', background: color + '15' }}>
                                    <p className="text-sm font-medium capitalize">{name}</p>
                                    <p className="text-xs mono text-white/60">{Math.round(conf * 100)}% confidence</p>
                                </div>
                            );
                        })}
                    </div>
                </motion.div>
            )}

            {annotationData && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass rounded-xl p-6"
                >
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <FileText className="w-5 h-5 text-horus-green" />
                        Ground Truth Annotations
                    </h3>
                    <p className="text-xs text-white/60 mb-3">{annotationFileName} — {annotationData.length} frames</p>
                </motion.div>
            )}
        </div>
    );
}
