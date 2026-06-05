import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, Activity, Radar, Camera, Cpu, AlertTriangle, Wifi, Thermometer } from 'lucide-react';
import HeroSection from './components/HeroSection';
import RadarMap from './components/RadarMap';
import ThreatDashboard from './components/ThreatDashboard';
import CameraDetection from './components/CameraDetection';
import SensorPanel from './components/SensorPanel';
import DeviceList from './components/DeviceList';
import Globe3D from './components/Globe3D';
import { connectBackend } from './api/horus';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [systemStatus, setSystemStatus] = useState({
    online: false,
    threatLevel: 'low',
    devicesDetected: 0,
    alertsCount: 0,
  });

  useEffect(() => {
    const socket = connectBackend(
      (data) => {
        const deviceArr = data?.devices || data || [];
        const droneCount = deviceArr.filter(d => d.type === 'drone').length;
        setSystemStatus(prev => ({
          ...prev,
          online: true,
          devicesDetected: deviceArr.length,
          threatLevel: droneCount > 0 ? 'high' : 'low',
          alertsCount: droneCount,
        }));
      },
      (sensors) => {
        if (sensors) {
          setSystemStatus(prev => ({ ...prev, online: true }));
        }
      }
    );
    return () => socket.disconnect();
  }, []);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'radar', label: 'Radar Map', icon: Radar },
    { id: 'camera', label: 'Camera AI', icon: Camera },
    { id: 'sensors', label: 'Sensors', icon: Activity },
    { id: 'devices', label: 'WiFi Devices', icon: Wifi },
  ];

  return (
    <div className="min-h-screen bg-horus-darker grid-bg">
      <div className="scan-line fixed inset-x-0 top-0 z-50 pointer-events-none" />

      <nav className="glass sticky top-0 z-40 px-6 py-4 border-b border-horus-cyan/20">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-3">
            <div className="relative">
              <Eye className="w-8 h-8 text-horus-cyan glow-text" />
              <span className="absolute inset-0 animate-pulse bg-horus-cyan/30 rounded-full blur-xl" />
            </div>
            <div>
              <h1 className="text-2xl font-bold gradient-text">HORUS EYE</h1>
              <p className="text-xs mono text-horus-cyan/60">v2.0 — AI + Multi-Sensor Fusion</p>
            </div>
          </motion.div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${systemStatus.online ? (systemStatus.threatLevel === 'high' ? 'threat-high' : 'threat-low') : 'bg-gray-500'}`} />
              <span className="text-sm mono text-horus-cyan/80">
                {systemStatus.online ? (systemStatus.threatLevel === 'high' ? 'THREAT DETECTED' : 'SYSTEM ONLINE') : 'OFFLINE'}
              </span>
            </div>
            <div className="text-sm mono">
              <span className="text-horus-cyan/60">DEVICES: </span>
              <span className="text-horus-cyan">{systemStatus.devicesDetected}</span>
            </div>
            <div className="text-sm mono">
              <span className="text-horus-cyan/60">DRONES: </span>
              <span className="text-horus-amber">{systemStatus.alertsCount}</span>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto mt-4 flex gap-2">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={`relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-all
                  ${activeTab === tab.id ? 'bg-horus-cyan/20 text-horus-cyan' : 'text-white/60 hover:text-horus-cyan hover:bg-white/5'}`}>
                <Icon className="w-4 h-4" />
                {tab.label}
                {activeTab === tab.id && (
                  <motion.div layoutId="active-tab" className="absolute inset-0 border border-horus-cyan/50 rounded-lg" />
                )}
              </button>
            );
          })}
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div key="overview" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <HeroSection />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
                <ThreatDashboard />
                <Globe3D />
              </div>
            </motion.div>
          )}
          {activeTab === 'radar' && <motion.div key="radar" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}><RadarMap /></motion.div>}
          {activeTab === 'camera' && <motion.div key="camera" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}><CameraDetection /></motion.div>}
          {activeTab === 'sensors' && <motion.div key="sensors" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}><SensorPanel /></motion.div>}
          {activeTab === 'devices' && <motion.div key="devices" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}><DeviceList /></motion.div>}
        </AnimatePresence>
      </main>

      <footer className="mt-16 py-8 border-t border-horus-cyan/10">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-sm text-horus-cyan/40 mono">
            HORUS EYE © 2026 — ESP32 + AI Classification + Real-time Web Dashboard
          </p>
        </div>
      </footer>
    </div>
  );
}
