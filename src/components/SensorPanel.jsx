import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Thermometer, Gauge, Activity, Radio, Mountain, AlertTriangle, Wind, Eye } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, LineChart, Line } from 'recharts';
import { connectBackend } from '../api/horus';

const MODE_COLORS = {
  NORMAL: '#00ff88', HEAT_WAVE: '#ff6b35', STORM: '#8866ff',
  THREAT_OVERHEAD: '#ff0055', COLD: '#00ccff', UNKNOWN: '#888'
};

const ZONE_LABELS = {
  NONE: 'No Detection', OUTER: 'Outer Zone >3m', INNER: 'Inner Zone <3m', PROXIMITY: 'Proximity <0.5m'
};

export default function SensorPanel() {
  const [data, setData] = useState({
    temperature: 0, pressure: null, altitude: null, vertical_velocity: null,
    bmp180_ok: false,
    pir_active: false, mode: 'NORMAL', zone: 'NONE', mock: false,
    radar_sky: false, radar_left: false, radar_right: false, radar_back: false,
    alert_active: false, device_count: 0
  });
  const [tempHistory, setTempHistory] = useState([]);
  const [pressureHistory, setPressureHistory] = useState([]);
  const [droneNearby, setDroneNearby] = useState(false);

  useEffect(() => {
    const socket = connectBackend(
      null,
      (sensorData) => {
        if (!sensorData) return;
        setData(prev => ({ ...prev, ...sensorData }));
        setTempHistory(prev => {
          const next = [...prev, { time: prev.length, temp: sensorData.temperature || 0 }];
          if (next.length > 60) next.shift();
          return next;
        });
        setPressureHistory(prev => {
          const next = [...prev, { time: prev.length, pressure: sensorData.pressure || 1013 }];
          if (next.length > 60) next.shift();
          return next;
        });
      }
    );
    return () => socket.disconnect();
  }, []);

  useEffect(() => {
    const handler = (st) => setDroneNearby(st.count > 0);
    window.addEventListener('droneStatus', handler);
    return () => window.removeEventListener('droneStatus', handler);
  }, []);

  const modeColor = MODE_COLORS[data.mode] || '#888';

  const bmpOk = data.bmp180_ok === true;
  const sensors = [
    { id: 'temp', icon: Thermometer, label: 'LM35 Temperature', value: data.temperature?.toFixed(1) || '--', unit: '°C', color: '#ff6b35' },
    { id: 'pir', icon: Activity, label: 'AM312 PIR', value: data.pir_active ? 'MOTION' : 'IDLE', unit: '', color: data.pir_active ? '#ff0055' : '#00ff88' },
    { id: 'pressure', icon: Gauge, label: 'BMP180 Pressure', value: bmpOk ? (data.pressure?.toFixed(1) || '--') : 'DISABLED', unit: bmpOk ? 'hPa' : '', color: bmpOk ? '#8866ff' : '#555' },
    { id: 'altitude', icon: Mountain, label: 'BMP180 Altitude', value: bmpOk ? (data.altitude?.toFixed(1) || '--') : 'DISABLED', unit: bmpOk ? 'm' : '', color: bmpOk ? '#00aaff' : '#555' },
    { id: 'vertical', icon: Wind, label: 'Vertical Velocity', value: bmpOk ? (data.vertical_velocity?.toFixed(2) || '--') : 'DISABLED', unit: bmpOk ? 'm/s' : '', color: bmpOk ? '#00ccff' : '#555' },
  ];

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 flex-wrap mb-2">
          <h2 className="text-3xl font-bold gradient-text">Sensor Telemetry</h2>
          {/* Mode badge */}
          <div className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-bold"
            style={{ background: `${modeColor}20`, border: `1px solid ${modeColor}50`, color: modeColor }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: modeColor }} />
            {data.mode}
          </div>
          {/* Zone badge */}
          <div className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-bold"
            style={{ background: `${modeColor}15`, border: `1px solid ${modeColor}40`, color: modeColor }}>
            <Eye className="w-3 h-3" />
            {ZONE_LABELS[data.zone] || data.zone}
          </div>
          {data.mock && (
            <div className="px-3 py-1 rounded-full bg-yellow-500/20 border border-yellow-500/50 text-yellow-400 text-xs font-bold">
              SIMULATED
            </div>
          )}
        </div>
        <p className="text-white/60 text-sm">
          Live readings from ESP32 — {data.device_count || 0} devices in range
          {droneNearby && <span className="text-horus-red font-bold ml-2 animate-pulse">⚠ Drone Alert Active</span>}
        </p>
      </motion.div>

      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {sensors.map((s, i) => {
          const Icon = s.icon;
          const isPir = s.id === 'pir';
          return (
            <motion.div key={s.id} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }} className="glass rounded-xl p-5">
              <div className="flex items-start justify-between mb-3">
                <Icon className="w-6 h-6" style={{ color: s.color }} />
                <div className={`w-2 h-2 rounded-full animate-pulse ${isPir && data.pir_active ? 'bg-horus-red' : 'bg-horus-green'}`} />
              </div>
              <p className="text-xs text-white/50 uppercase mono mb-2">{s.label}</p>
              <p className="text-2xl xl:text-3xl font-bold mono">{s.value}<span className="text-sm text-white/40 ml-1">{s.unit}</span></p>
            </motion.div>
          );
        })}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }} className="glass rounded-xl p-5">
          <div className="flex items-start justify-between mb-3">
            <Radio className="w-6 h-6" style={{ color: '#00aaff' }} />
            <div className={`w-2 h-2 rounded-full animate-pulse ${data.alert_active ? 'bg-horus-red' : 'bg-horus-green'}`} />
          </div>
          <p className="text-xs text-white/50 uppercase mono mb-2">System Status</p>
          <p className="text-lg font-bold mono" style={{ color: data.alert_active ? '#ff0055' : '#00ff88' }}>
            {data.alert_active ? 'ALERT' : 'CLEAR'}
          </p>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Radio className="w-5 h-5 text-horus-cyan" />
            RCWL-0516 Radar Array
          </h3>
          <div className="relative aspect-square max-w-sm mx-auto">
            <div className="absolute inset-0 rounded-full border-2" style={{ borderColor: `${modeColor}30` }} />
            <div className="absolute inset-[20%] rounded-full border" style={{ borderColor: `${modeColor}20` }} />
            {[
              { pos: 'top', key: 'radar_sky', label: 'SKY', x: '50%', y: '5%' },
              { pos: 'left', key: 'radar_left', label: 'WEST', x: '5%', y: '50%' },
              { pos: 'right', key: 'radar_right', label: 'EAST', x: '95%', y: '50%' },
              { pos: 'bottom', key: 'radar_back', label: 'BACK', x: '50%', y: '95%' },
            ].map(s => (
              <div key={s.key} className="absolute" style={{ left: s.x, top: s.y, transform: 'translate(-50%, -50%)' }}>
                <div className={`relative w-14 h-14 rounded-full flex items-center justify-center transition-all
                  ${data[s.key] ? 'border-2 shadow-lg' : 'bg-horus-cyan/10 border-2 border-horus-cyan/40'}`}
                  style={data[s.key] ? { background: `${modeColor}20`, borderColor: modeColor, boxShadow: `0 0 20px ${modeColor}50` } : {}}>
                  {data[s.key] && <div className="absolute inset-0 rounded-full animate-ping" style={{ background: `${modeColor}30` }} />}
                  <span className="mono text-xs font-bold">{s.label}</span>
                </div>
              </div>
            ))}
            <div className="absolute inset-[40%] rounded-full flex items-center justify-center" style={{ background: `${modeColor}30` }}>
              <span className="mono text-xs" style={{ color: modeColor }}>ESP32</span>
            </div>
          </div>
        </motion.div>

        <div className="space-y-4">
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Thermometer className="w-5 h-5 text-horus-cyan" />
              LM35 Temperature History
            </h3>
            <div className="h-48">
              {tempHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={tempHistory}>
                    <defs><linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff6b35" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#ff6b35" stopOpacity={0} />
                    </linearGradient></defs>
                    <XAxis dataKey="time" stroke="#666" fontSize={10} />
                    <YAxis stroke="#666" fontSize={10} />
                    <Area type="monotone" dataKey="temp" stroke="#ff6b35" strokeWidth={2} fill="url(#tempGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : <div className="flex items-center justify-center h-full text-white/40">Waiting...</div>}
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="glass rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Gauge className="w-5 h-5 text-horus-cyan" />
              BMP180 Pressure History
            </h3>
            <div className="h-48">
              {pressureHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={pressureHistory}>
                    <defs><linearGradient id="pressGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8866ff" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#8866ff" stopOpacity={0} />
                    </linearGradient></defs>
                    <XAxis dataKey="time" stroke="#666" fontSize={10} />
                    <YAxis stroke="#666" fontSize={10} domain={['auto', 'auto']} />
                    <Area type="monotone" dataKey="pressure" stroke="#8866ff" strokeWidth={2} fill="url(#pressGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : <div className="flex items-center justify-center h-full text-white/40">Waiting...</div>}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}