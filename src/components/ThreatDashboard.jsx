import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, Activity, Zap, Bell, Clock } from 'lucide-react';
import { connectBackend, onDroneAlert } from '../api/horus';

export default function ThreatDashboard() {
  const [droneCount, setDroneCount] = useState(0);
  const [totalDevices, setTotalDevices] = useState(0);
  const [alertHistory, setAlertHistory] = useState([]);
  const [lastAlertTime, setLastAlertTime] = useState(null);

  useEffect(() => {
    const socket = connectBackend((data) => {
      const deviceArr = data?.devices || data || [];
      if (!deviceArr.length && !data) return;
      setDroneCount(deviceArr.filter(d => d.type === 'drone').length);
      setTotalDevices(deviceArr.length);
    });
    return () => socket.disconnect();
  }, []);

  // Listen for real-time drone alerts
  useEffect(() => {
    const handler = (data) => {
      setAlertHistory(prev => [...data.drones, ...prev].slice(0, 10));
      setLastAlertTime(new Date());
    };
    window.addEventListener('droneAlert', handler);
    return () => window.removeEventListener('droneAlert', handler);
  }, []);

  const threats = [
    { level: 'HIGH', count: droneCount, color: '#ff0055', icon: AlertTriangle },
    { level: 'MEDIUM', count: Math.floor(totalDevices * 0.1), color: '#ffaa00', icon: Activity },
    { level: 'LOW', count: totalDevices - droneCount, color: '#00ff88', icon: Shield },
  ];

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-xl p-6">
      <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
        <Zap className="w-5 h-5 text-horus-cyan" />
        Threat Dashboard
      </h3>

      <div className="space-y-4">
        {threats.map((t, i) => {
          const Icon = t.icon;
          return (
            <motion.div key={t.level} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
              className="flex items-center justify-between p-4 rounded-lg border" style={{ borderColor: `${t.color}40`, background: `${t.color}10` }}>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: `${t.color}30` }}>
                  <Icon className="w-5 h-5" style={{ color: t.color }} />
                </div>
                <div>
                  <p className="font-medium mono">{t.level}</p>
                  <p className="text-xs text-white/60">Threat Level</p>
                </div>
              </div>
              <div className="text-3xl font-bold mono" style={{ color: t.color }}>
                {t.count}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Active drone alert */}
      {droneCount > 0 && (
        <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} className="mt-4 p-4 rounded-lg bg-horus-red/10 border border-horus-red/50">
          <p className="text-horus-red text-sm font-bold mono flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 animate-pulse" />
            {droneCount} DRONE{droneCount > 1 ? 'S' : ''} DETECTED IN AIRSPACE
          </p>
          {lastAlertTime && (
            <p className="text-xs text-horus-red/60 mt-1 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              Last alert: {lastAlertTime.toLocaleTimeString()}
            </p>
          )}
        </motion.div>
      )}

      {/* Recent alert history */}
      {alertHistory.length > 0 && (
        <div className="mt-4 pt-4 border-t border-horus-cyan/10">
          <p className="text-xs text-white/60 mb-2 mono uppercase">Recent Alerts</p>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {alertHistory.map((alert, i) => (
              <div key={i} className="flex items-center justify-between p-2 rounded bg-white/5 text-xs">
                <div className="flex items-center gap-2">
                  <span className="text-horus-red">
                    <AlertTriangle className="w-3 h-3" />
                  </span>
                  <span className="mono text-white/80">{alert.vendor || 'Unknown'}</span>
                  <span className="mono text-white/40">{alert.mac.substring(0, 8)}..</span>
                </div>
                <span className="text-white/40 mono text-[10px]">
                  {alert.method?.replace(/_/g, ' ')} · {(alert.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 pt-6 border-t border-horus-cyan/10">
        <p className="text-xs text-white/50 mb-3 mono uppercase">Response Relays (RM6W)</p>
        <div className="grid grid-cols-6 gap-2">
          {['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6'].map((ch, i) => (
            <div key={ch} className={`text-center p-2 rounded text-xs mono ${i < droneCount && droneCount > 0 ? 'bg-horus-red/20 text-horus-red' : 'bg-horus-cyan/10 text-horus-cyan/60'}`}>
              {ch}
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}