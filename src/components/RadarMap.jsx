import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Shield, Eye, Search, Download, Mountain } from 'lucide-react';
import { connectBackend, onDroneAlert, horusAPI } from '../api/horus';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

// ============================================================
// RSSI-based distance estimation using log-distance path loss model
// Calibrate RSSI_AT_1M for your specific ESP32 antenna/hardware
// ============================================================

const RSSI_AT_1M = -43;
const PATH_LOSS_N = 2.8;

function estimateDistance(rssi) {
  if (!rssi || rssi >= -10) return 0.3;
  if (rssi < -100) return 150;
  const d = Math.pow(10, (RSSI_AT_1M - rssi) / (10 * PATH_LOSS_N));
  return Math.max(0.3, Math.min(150, d));
}

function formatDistance(m) {
  if (m < 1) return m.toFixed(1) + ' m';
  if (m < 1000) return m.toFixed(1) + ' m';
  return (m / 1000).toFixed(2) + ' km';
}

function macToAngle(mac) {
  let hash = 0;
  for (let i = 0; i < mac.length; i++) {
    hash = ((hash << 5) - hash) + mac.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash % 360);
}

const COLOR_MAP = {
  drone: '#ff0055', phone: '#00aaff', router: '#00ff88',
  laptop: '#ffaa00', iot: '#8888ff', other: '#888'
};

// ============================================================
// TOAST NOTIFICATION SYSTEM
// ============================================================

function Toast({ toasts, onDismiss }) {
  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 pointer-events-none">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 100 }}
            className="pointer-events-auto glass rounded-xl p-4 border-l-4 max-w-sm shadow-2xl"
            style={{ borderLeftColor: toast.color }}
          >
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" style={{ color: toast.color }} />
              <div>
                <p className="font-bold text-sm" style={{ color: toast.color }}>{toast.title}</p>
                <p className="text-xs text-white/60 mt-0.5">{toast.message}</p>
                <p className="text-[10px] text-white/30 mt-1">{toast.time}</p>
              </div>
              <button onClick={() => onDismiss(toast.id)} className="text-white/30 hover:text-white ml-2">×</button>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

// ============================================================
// COMPONENT
// ============================================================

export default function RadarMap() {
  const [devices, setDevices] = useState([]);
  const [sensorData, setSensorData] = useState({ mode: 'NORMAL', zone: 'NONE', altitude: 0 });
  const [isMock, setIsMock] = useState(false);
  const [scanning, setScanning] = useState(true);
  const [toasts, setToasts] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState('3d');
  const sweepRef = useRef(0);
  const toastIdRef = useRef(0);

  const MODE_COLORS = {
    NORMAL: '#00ff88', HEAT_WAVE: '#ff6b35', STORM: '#8866ff',
    THREAT_OVERHEAD: '#ff0055', COLD: '#00ccff', UNKNOWN: '#888'
  };

  const ZONE_COLORS = {
    NONE: '#444', OUTER: '#00aaff', INNER: '#ffaa00', PROXIMITY: '#ff0055', UNKNOWN: '#888'
  };

  // Real-time data connection via backend
  useEffect(() => {
    const socket = connectBackend(
      (data) => {
        if (!data) return;
        const devicesArr = data.devices || data;
        const mockFlag = data.mock || false;
        setIsMock(mockFlag);
        const enriched = devicesArr.map(d => ({
          ...d,
          estimatedDist: estimateDistance(d.rssi || d.avg_rssi || -80)
        }));
        setDevices(enriched);
      },
      (data) => {
        if (!data) return;
        setIsMock(data.mock || false);
        setSensorData(prev => ({
          ...prev,
          mode: data.mode || prev.mode,
          zone: data.zone || prev.zone,
          altitude: data.altitude || prev.altitude,
          pressure: data.pressure || prev.pressure,
          vertical_velocity: data.vertical_velocity || prev.vertical_velocity
        }));
      }
    );
    setScanning(true);
    return () => { socket.disconnect(); setScanning(false); };
  }, []);

  // Drone alert toast notifications
  useEffect(() => {
    return onDroneAlert((data) => {
      const { drones } = data;
      drones.forEach(d => {
        const id = ++toastIdRef.current;
        setToasts(prev => [
          {
            id,
            title: `🚨 DRONE: ${d.vendor || 'Unknown'}`,
            message: `${d.mac} — ${d.method} (${Math.round(d.confidence * 100)}%) — Est. ${formatDistance(d.estimatedDist || estimateDistance(d.rssi))}`,
            color: '#ff0055',
            time: new Date().toLocaleTimeString()
          },
          ...prev.slice(0, 4)
        ]);
        // Auto-dismiss after 8 seconds
        setTimeout(() => {
          setToasts(prev => prev.filter(t => t.id !== id));
        }, 8000);
      });
    });
  }, []);

  // Sweep rotation animation
  useEffect(() => {
    const id = setInterval(() => {
      sweepRef.current = (sweepRef.current + 1.2) % 360;
    }, 30);
    return () => clearInterval(id);
  }, []);

  const sweepDeg = sweepRef.current;
  const droneCount = devices.filter(d => d.type === 'drone').length;

  const MAX_DIST_M = 50;
  const radiusPct = (d) => Math.min(45, (d / MAX_DIST_M) * 45);

  // Apply search filter
  const filteredDevices = searchQuery
    ? devices.filter(d =>
        d.mac.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (d.vendor || '').toLowerCase().includes(searchQuery.toLowerCase())
      )
    : devices;

const counts = {};
  devices.forEach(d => { counts[d.type] = (counts[d.type] || 0) + 1; });

  const distStats = {};
  devices.forEach(d => {
    const dist = d.estimatedDist || 0;
    if (!distStats[d.type]) distStats[d.type] = [];
    distStats[d.type].push(dist);
  });
  for (const [type, arr] of Object.entries(distStats)) {
    const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
    distStats[type] = { avg, min: Math.min(...arr), max: Math.max(...arr) };
  }

  const handleExport = () => {
    const url = horusAPI.exportCSV();
    const a = document.createElement('a');
    a.href = url;
    a.download = `horus-devices-${Date.now()}.csv`;
    a.click();
  };

  return (
    <div className="space-y-6">
      <Toast toasts={toasts} onDismiss={(id) => setToasts(prev => prev.filter(t => t.id !== id))} />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-2xl p-6"
        style={{ borderColor: `${MODE_COLORS[sensorData.mode] || '#444'}30`, borderWidth: 1 }}
      >
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <div>
            <h2 className="text-3xl font-bold gradient-text mb-1">360° Radar Sweep</h2>
            <p className="text-white/60 text-sm">
              RSSI distance estimation · {filteredDevices.length} devices
              {searchQuery && ' · filtered'}
              {isMock && <span className="text-yellow-400 font-bold ml-2">[SIMULATED]</span>}
            </p>
          </div>
          <div className="flex items-center gap-4 flex-wrap">
            {/* Mode badge */}
            <div className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-bold"
              style={{ background: `${MODE_COLORS[sensorData.mode] || '#444'}20`, border: `1px solid ${MODE_COLORS[sensorData.mode] || '#444'}50`, color: MODE_COLORS[sensorData.mode] || '#888' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: MODE_COLORS[sensorData.mode] }} />
              {sensorData.mode}
            </div>

            {/* Zone badge */}
            <div className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-bold"
              style={{ background: `${ZONE_COLORS[sensorData.zone] || '#444'}20`, border: `1px solid ${ZONE_COLORS[sensorData.zone] || '#444'}50`, color: ZONE_COLORS[sensorData.zone] || '#888' }}>
              <span>ZONE:</span>
              <span>{sensorData.zone}</span>
            </div>

            {/* View mode toggle */}
            <button onClick={() => setViewMode(v => v === '2d' ? '3d' : '2d')}
              className="px-3 py-1 rounded-lg border text-xs font-bold transition-colors"
              style={{ borderColor: viewMode === '3d' ? '#00ff88' : '#444', color: viewMode === '3d' ? '#00ff88' : '#888' }}>
              {viewMode === '3d' ? '3D' : '2D'}
            </button>

            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
              <input type="text" placeholder="Search MAC or vendor..." value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="pl-9 pr-4 py-1.5 bg-white/5 border border-white/10 rounded-lg text-sm w-48 focus:outline-none focus:border-horus-cyan/40" />
            </div>

            {/* Live indicator */}
            <div className="flex items-center gap-2 mono text-xs">
              <div className={`w-2 h-2 rounded-full ${scanning ? 'threat-low' : 'bg-gray-600'}`} />
              <span className={scanning ? 'text-horus-cyan' : 'text-white/40'}>{scanning ? 'LIVE' : 'PAUSED'}</span>
            </div>

            {/* Drone count badge */}
            {droneCount > 0 && (
              <div className="flex items-center gap-1 px-3 py-1 rounded-full bg-horus-red/20 border border-horus-red/50 text-horus-red text-xs font-bold animate-pulse">
                <AlertTriangle className="w-3 h-3" />
                {droneCount} DRONE{droneCount > 1 ? 'S' : ''}
              </div>
            )}

            {/* Export CSV */}
            <button onClick={handleExport}
              className="flex items-center gap-1 px-3 py-1 rounded-lg bg-white/5 border border-white/10 text-xs text-white/60 hover:text-horus-cyan hover:border-horus-cyan/30 transition-colors">
              <Download className="w-3 h-3" /> CSV
            </button>
          </div>
        </div>

        {/* Radar view: 2D or 3D */}
        <div className="relative w-full max-w-md mx-auto" style={{ paddingBottom: '100%' }}>
          {viewMode === '2d' ? (
          <div className="absolute inset-0">
            {/* Zone rings — Outer (>3m), Inner (<3m), Proximity (<0.5m) */}
            {[
              { m: 50, label: '50m', color: 'rgba(0,170,255,0.04)', border: 'rgba(0,170,255,0.15)' },
              { m: 10, label: '10m', color: 'rgba(0,170,255,0.06)', border: 'rgba(0,170,255,0.12)' },
              { m: 3, label: '3m Outer', color: 'rgba(0,170,255,0.08)', border: 'rgba(0,170,255,0.25)' },
              { m: 0.5, label: '0.5m Inner', color: 'rgba(255,170,0,0.08)', border: 'rgba(255,170,0,0.25)' },
            ].map((ring) => {
              const pct = radiusPct(ring.m);
              const isActive = (ring.m === 0.5 && (sensorData.zone === 'PROXIMITY' || sensorData.zone === 'INNER')) ||
                               (ring.m === 3 && (sensorData.zone === 'OUTER' || sensorData.zone === 'INNER' || sensorData.zone === 'PROXIMITY'));
              return (
                <div key={ring.m} className="absolute rounded-full transition-all duration-500"
                  style={{
                    width: `${pct * 2}%`, height: `${pct * 2}%`,
                    left: `${50 - pct}%`, top: `${50 - pct}%`,
                    background: isActive ? ring.color : 'transparent',
                    border: `1px solid ${ring.border}`,
                    boxShadow: isActive ? `0 0 20px ${ring.border}` : 'none'
                  }}>
                  <div className="absolute -top-5 left-1/2 -translate-x-1/2 mono text-[9px]"
                    style={{ color: isActive ? ring.border : 'rgba(0,255,200,0.3)' }}>{ring.label}</div>
                </div>
              );
            })}

            {/* Crosshairs */}
            <div className="absolute top-1/2 left-0 right-0 h-px" style={{ background: `${MODE_COLORS[sensorData.mode] || '#888'}15` }} />
            <div className="absolute left-1/2 top-0 bottom-0 w-px" style={{ background: `${MODE_COLORS[sensorData.mode] || '#888'}15` }} />

            {/* Direction labels */}
            {[['N',0],['NE',45],['E',90],['SE',135],['S',180],['SW',225],['W',270],['NW',315]].map(([dir,angle]) => {
              const rad = (angle * Math.PI) / 180;
              return (
                <div key={dir} className="absolute mono text-[10px] whitespace-nowrap"
                  style={{ top: `${50 - 49 * Math.cos(rad)}%`, left: `${50 + 49 * Math.sin(rad)}%`, transform: 'translate(-50%,-50%)', color: `${MODE_COLORS[sensorData.mode] || '#888'}40` }}>{dir}</div>);
            })}

            {/* Sweep beam */}
            <div className="absolute top-1/2 left-1/2 origin-left h-[2px]"
              style={{
                width: '45%', transform: `rotate(${sweepDeg}deg)`,
                background: `linear-gradient(to right, ${MODE_COLORS[sensorData.mode] || '#00ff88'}99, ${MODE_COLORS[sensorData.mode] || '#00ff88'}15, transparent)`,
                filter: 'blur(2px)',
              }} />

            {/* Center */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full z-20 animate-pulse"
              style={{ background: MODE_COLORS[sensorData.mode] || '#00ff88', boxShadow: `0 0 12px ${MODE_COLORS[sensorData.mode] || '#00ff88'}80` }} />

            {/* Device markers */}
            <AnimatePresence>
              {filteredDevices.map((device, idx) => {
                const dist = device.estimatedDist || 0;
                const angle = macToAngle(device.mac || '');
                const rad = (angle * Math.PI) / 180;
                const pct = radiusPct(dist);
                const color = COLOR_MAP[device.type] || '#888';
                const x = 50 + pct * Math.sin(rad);
                const y = 50 - pct * Math.cos(rad);

                return (
                  <motion.div key={device.mac}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    transition={{ delay: idx * 0.03, duration: 0.4 }}
                    className="absolute cursor-pointer group z-10"
                    style={{ left: `${x}%`, top: `${y}%`, transform: 'translate(-50%,-50%)' }}>
                    {device.type === 'drone' && (
                      <div className="absolute -inset-[10px] rounded-full animate-ping" style={{ background: color, opacity: 0.2 }} />
                    )}
                    <div className="relative w-4 h-4 rounded-full shadow-lg" style={{ background: color, boxShadow: `0 0 8px ${color}80` }}>
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-30 w-max">
                        <div className="glass px-3 py-2 rounded-lg text-xs border border-white/10 min-w-[260px]">
                          <p className="mono font-bold text-horus-cyan mb-1">{device.mac}</p>
                          <p className="text-white/60 text-[11px] mb-1">{device.vendor || 'Unknown'}</p>
                          <div className="flex justify-between items-center mb-0.5 gap-2">
                            <span className="capitalize" style={{ color, fontWeight: 600 }}>{device.type} ({Math.round(device.confidence * 100)}%)</span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded capitalize" style={{ background: `${color}20`, color }}>{device.method?.replace(/_/g, ' ')}</span>
                          </div>
                          <div className="flex justify-between mt-0.5 text-[10px] text-white/50">
                            <span>RSSI: <span className="text-white/80">{device.rssi} dBm</span></span>
                            <span>CH: <span className="text-white/80">{device.channel}</span></span>
                            <span>PKTs: <span className="text-white/80">{device.packets}</span></span>
                          </div>
                          <p className="mono text-[10px] text-white/50 mt-1">Est. distance: <span className="font-bold text-horus-cyan">{formatDistance(dist)}</span></p>
                          {device.type === 'drone' && <p className="text-horus-red text-[10px] mt-1">⚠ Drone classification: OUI-confirmed, high confidence</p>}
                        </div>
                      </div>
                    </div>
                    <div className="absolute bottom-[-16px] left-1/2 -translate-x-1/2 whitespace-nowrap">
                      <span className="mono text-[7px] opacity-0">{formatDistance(dist)}</span>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {/* Legend */}
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-3 mono text-[9px] text-white/40">
              {['drone','phone','router','laptop','iot'].map(type =>
                <span key={type} className="flex items-center gap-1" style={{ fontWeight: counts[type] ? 600 : 400 }}>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: COLOR_MAP[type] }} />
                  <span className="capitalize">{type}</span>
                  <span style={{ color: COLOR_MAP[type] }}>{counts[type] || 0}</span>
                </span>
              )}
            </div>
          </div>
          ) : (
          <div className="absolute inset-0 rounded-2xl overflow-hidden">
            <RadarScene3D
              devices={filteredDevices}
              mode={sensorData.mode}
              zone={sensorData.zone}
              altitude={sensorData.altitude}
              modeColor={MODE_COLORS[sensorData.mode] || '#00ff88'}
            />
          </div>
          )}
        </div>

        {/* Altitude bar (2D only) */}
        {viewMode === '2d' && sensorData.altitude > 0 && (
          <div className="mt-4 flex items-center gap-3 glass rounded-lg px-4 py-2">
            <Mountain className="w-4 h-4 text-horus-cyan" />
            <div className="flex-1 h-2 rounded-full bg-white/5 overflow-hidden">
              <div className="h-full rounded-full transition-all duration-1000"
                style={{ width: `${Math.min(100, sensorData.altitude / 5)}%`, background: `linear-gradient(to right, #00ff88, ${MODE_COLORS[sensorData.mode] || '#00ff88'})` }} />
            </div>
            <span className="mono text-xs text-horus-cyan">{sensorData.altitude.toFixed(1)} m</span>
          </div>
        )}

        {/* Distance stats cards */}
        <AnimatePresence>
          {(devices.length > 0) && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-6">
              {Object.entries(distStats).map(([type, stats], i) => (
                <motion.div key={type}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="glass rounded-lg p-3 text-center border"
                  style={{ borderColor: `${COLOR_MAP[type]}30` }}
                >
                  <p className="uppercase mono text-[9px] mb-1" style={{ color: COLOR_MAP[type] }}>
                    {type} ({counts[type] || 0})
                  </p>
                  <p className="mono text-sm" style={{ color: COLOR_MAP[type] }}>{formatDistance(stats.avg)}</p>
                  <p className="mono text-[10px] text-white/40 mt-1">
                    range: {formatDistance(stats.min)} — {formatDistance(stats.max)}
                  </p>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

// ==================== Three.js 3D Radar Scene ====================

const RING_SEGMENTS = 64;

function genRingPts(radius) {
  const pts = [];
  for (let i = 0; i <= RING_SEGMENTS; i++) {
    const theta = (i / RING_SEGMENTS) * Math.PI * 2;
    pts.push([Math.cos(theta) * radius, 0, Math.sin(theta) * radius]);
  }
  return pts;
}

function DeviceMarker({ device }) {
  const meshRef = useRef();
  const dist = device.estimatedDist || 1;
  const angle = macToAngle(device.mac || '');
  const rad = (angle * Math.PI) / 180;
  const typeColor = COLOR_MAP[device.type] || '#888';
  const isDrone = device.type === 'drone';
  const r = Math.min(dist / 50, 1) * 4;
  const x = r * Math.sin(rad);
  const z = r * Math.cos(rad);
  const y = isDrone ? 0.5 : 0.1;

  useFrame((state) => {
    if (meshRef.current && isDrone) {
      meshRef.current.position.y = 0.5 + Math.sin(state.clock.elapsedTime * 2 + dist) * 0.15;
    }
  });

  return (
    <group ref={meshRef} position={[x, y, z]}>
      <mesh>
        <sphereGeometry args={[isDrone ? 0.12 : 0.08, 16, 16]} />
        <meshStandardMaterial color={typeColor} emissive={typeColor} emissiveIntensity={isDrone ? 0.8 : 0.3} />
      </mesh>
      {isDrone && (
        <mesh>
          <ringGeometry args={[0.15, 0.25, 32]} />
          <meshBasicMaterial color={typeColor} transparent opacity={0.4} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  );
}

function SceneContent({ devices, zone, altitude, modeColor }) {
  const groupRef = useRef();

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.001;
    }
  });

  const rings = useMemo(() => [
    { radius: 4, color: modeColor, opacity: 0.12 },
    { radius: 2.5, color: '#00aaff', opacity: zone === 'OUTER' || zone === 'INNER' || zone === 'PROXIMITY' ? 0.6 : 0.12 },
    { radius: 0.8, color: '#ffaa00', opacity: zone === 'INNER' || zone === 'PROXIMITY' ? 0.7 : 0.12 },
    { radius: 0.2, color: '#ff0055', opacity: zone === 'PROXIMITY' ? 0.9 : 0.12 },
  ], [zone, modeColor]);

  return (
    <group ref={groupRef}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
        <planeGeometry args={[12, 12]} />
        <meshStandardMaterial color="#0a0a1a" transparent opacity={0.8} />
      </mesh>

      {rings.map((ring, i) => (
        <Line key={i} points={genRingPts(ring.radius)} color={ring.color} lineWidth={1} transparent opacity={ring.opacity} />
      ))}

      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshStandardMaterial color={modeColor} emissive={modeColor} emissiveIntensity={0.5} />
      </mesh>

      {devices.map((device) => (
        <DeviceMarker key={device.mac} device={device} />
      ))}

      {altitude > 0 && (
        <Line points={[[0, 0, 0], [0, Math.min(altitude / 10, 3), 0]]} color={modeColor} lineWidth={2} transparent opacity={0.4} />
      )}

      <OrbitControls enablePan={false} minDistance={3} maxDistance={15} autoRotate autoRotateSpeed={0.5} />
    </group>
  );
}

function RadarScene3D({ devices, zone, altitude, modeColor }) {
  return (
    <Canvas camera={{ position: [6, 4, 6], fov: 50 }} gl={{ antialias: true }}>
      <ambientLight intensity={0.3} color={modeColor} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} color={modeColor} />
      <pointLight position={[0, 3, 0]} intensity={0.5} color={modeColor} />
      <fog attach="fog" args={[0x0a0a1a, 8, 20]} />
      <SceneContent devices={devices} zone={zone} altitude={altitude} modeColor={modeColor} />
    </Canvas>
  );
}