import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Wifi, Smartphone, Router, Laptop, AlertTriangle, Search, Bell } from 'lucide-react';
import { connectBackend, onDroneAlert, horusAPI } from '../api/horus';

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
  if (m < 100) return m.toFixed(1) + ' m';
  return (m / 1000).toFixed(2) + ' km';
}

const ICONS = { phone: Smartphone, router: Router, laptop: Laptop, drone: AlertTriangle, iot: Wifi, other: Wifi };
const COLORS = { phone: '#00aaff', router: '#00ff88', laptop: '#ffaa00', drone: '#ff0055', iot: '#8888ff', other: '#888' };

const METHOD_BADGES = {
  'drone_oui': { label: '🚨 Drone OUI', bg: '#ff005530', color: '#ff4466' },
  'confirmed_drone_oui': { label: '🚨 Confirmed Drone', bg: '#ff005530', color: '#ff4466' },
  'agent_confirmed_drone': { label: '🤖 Agent: Drone', bg: '#ff005530', color: '#ff4466' },
  'local_oui_db': { label: '✅ OUI Match', bg: '#00aaff20', color: '#00ccff' },
  'api_identified': { label: '🔍 API Found', bg: '#22aa8820', color: '#22dd88' },
  'randomized_mac': { label: '🔄 Random MAC', bg: '#ffaa0020', color: '#ffaa00' },
  'feature_analysis': { label: '📊 Feature AI', bg: '#88882020', color: '#aaa' },
  'insufficient_data': { label: '⚪ Low Data', bg: '#55555520', color: '#777' },
};

function getMethodBadge(method) {
  return METHOD_BADGES[method] || METHOD_BADGES['feature_analysis'];
}

export default function DeviceList() {
  const [devices, setDevices] = useState([]);
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [droneAlerts, setDroneAlerts] = useState([]);
  const [alertVisible, setAlertVisible] = useState(false);

  useEffect(() => {
    const socket = connectBackend((data) => {
      const deviceArr = data?.devices || data || [];
      setDevices(deviceArr);
    });
    return () => socket.disconnect();
  }, []);

  // Listen for drone alert events
  useEffect(() => {
    const handler = (data) => {
      setDroneAlerts(prev => [...data.drones, ...prev].slice(0, 20));
      setAlertVisible(true);
      setTimeout(() => setAlertVisible(false), 5000);
    };
    window.addEventListener('droneAlert', handler);
    return () => window.removeEventListener('droneAlert', handler);
  }, []);

  const filtered = devices.filter(d => {
    const matchesFilter = filter === 'all' || d.type === filter;
    const matchesSearch = d.mac.toLowerCase().includes(search.toLowerCase()) ||
      (d.vendor || '').toLowerCase().includes(search.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const threatCount = devices.filter(d => d.type === 'drone').length;

  const exportCSV = () => {
    const url = horusAPI.exportCSV();
    const a = document.createElement('a');
    a.href = url;
    a.download = `horus-devices-${Date.now()}.csv`;
    a.click();
  };

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-xl p-6">
        <div className="flex items-center justify-between mb-2 flex-wrap gap-3">
          <div>
            <h2 className="text-3xl font-bold gradient-text mb-2">WiFi Device Inventory</h2>
            <p className="text-white/60 text-sm">Classified in real-time by Horus AI ({devices.length} devices)</p>
          </div>
          <div className="flex items-center gap-3">
            {alertVisible && threatCount > 0 && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-horus-red/20 border border-horus-red/50"
              >
                <AlertTriangle className="w-5 h-5 text-horus-red animate-pulse" />
                <span className="text-horus-red font-bold mono">{threatCount} Drone{threatCount > 1 ? 's' : ''}!</span>
              </motion.div>
            )}
            <button
              onClick={exportCSV}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-xs text-white/60 hover:text-horus-cyan hover:border-horus-cyan/30 transition-colors"
            >
              <Bell className="w-3 h-3" />
              Export CSV
            </button>
          </div>
        </div>

        <div className="flex gap-4 mb-6 flex-wrap">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
            <input type="text" placeholder="Search MAC, vendor, or type..."
              value={search} onChange={e => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-white/5 border border-horus-cyan/20 rounded-lg text-sm focus:outline-none focus:border-horus-cyan/60" />
          </div>
          <div className="flex gap-2 flex-wrap">
            {['all', 'drone', 'phone', 'router', 'laptop', 'iot', 'other'].map(f => (
              <button key={f} onClick={() => setFilter(f)}
                className={`px-3 py-2 rounded-lg text-xs capitalize transition-all ${filter === f ? 'bg-horus-cyan text-black' : 'border border-horus-cyan/30 text-horus-cyan'}`}>
                {f}
              </button>
            ))}
          </div>
        </div>

        {/* Recent drone alerts banner */}
        {droneAlerts.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-3 rounded-lg bg-horus-red/10 border border-horus-red/30"
          >
            <p className="text-xs text-horus-red font-bold mb-1">⚠️ Recent Drone Alerts</p>
            <div className="flex flex-wrap gap-2">
              {droneAlerts.slice(0, 5).map((a, i) => (
                <span key={i} className="text-[10px] mono text-white/60 bg-white/5 px-2 py-1 rounded">
                  {a.vendor} ({a.mac.substring(0, 8)}...) — {a.method?.replace(/_/g, ' ')}
                </span>
              ))}
              {droneAlerts.length > 5 && (
                <span className="text-[10px] text-white/30">+{droneAlerts.length - 5} more</span>
              )}
            </div>
          </motion.div>
        )}

        <div className="space-y-2 max-h-[600px] overflow-y-auto">
          {filtered.map((d, i) => {
            const Icon = ICONS[d.type] || Wifi;
            const dist = estimateDistance(d.rssi);
            const badge = getMethodBadge(d.method);

            return (
              <motion.div key={d.mac}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.02 }}
                className={`flex flex-col lg:flex-row items-start lg:items-center justify-between p-4 rounded-lg transition-all gap-3 ${
                  d.type === 'drone' ? 'bg-horus-red/5 border border-horus-red/20' : 'bg-white/5 hover:bg-white/10'
                }`}
              >
                <div className="flex items-center gap-4 min-w-0">
                  <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: `${COLORS[d.type]}20` }}>
                    <Icon className="w-5 h-5" style={{ color: COLORS[d.type] }} />
                  </div>
                  <div className="min-w-0">
                    <p className={`font-medium mono text-sm truncate ${d.type === 'drone' ? 'text-horus-red' : ''}`}>
                      {d.mac}
                      {d.type === 'drone' && <span className="ml-2 text-[10px]">🚨</span>}
                    </p>
                    <p className="text-xs text-white/60">{d.vendor || 'Unknown'}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  <span className="text-[10px] px-2 py-1 rounded uppercase font-mono"
                    style={{ background: badge.bg, color: badge.color, whiteSpace: 'nowrap' }}>
                    {badge.label}
                  </span>
                  <span className="text-xs px-2 py-1 rounded capitalize"
                    style={{ background: `${COLORS[d.type]}20`, color: COLORS[d.type], whiteSpace: 'nowrap' }}>
                    {d.type} {(d.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="flex items-center gap-5 text-sm flex-shrink-0">
                  <div className="text-right mono">
                    <p className="text-white/50 text-[10px]">RSSI</p>
                    <p style={{ color: d.rssi > -60 ? '#00ff88' : d.rssi > -80 ? '#ffaa00' : '#ff5555' }}>
                      {d.rssi} dBm
                    </p>
                  </div>
                  <div className="text-right mono">
                    <p className="text-white/50 text-[10px]">Distance</p>
                    <p style={{ color: dist < 10 ? '#ff0055' : dist < 30 ? '#ffaa00' : '#00ff88' }}>
                      {formatDistance(dist)}
                    </p>
                  </div>
                  <div className="text-right mono">
                    <p className="text-white/50 text-[10px]">CH</p>
                    <p>{d.channel}</p>
                  </div>
                  <div className="text-right mono">
                    <p className="text-white/50 text-[10px]">PKTs</p>
                    <p>{d.packets}</p>
                  </div>
                </div>
              </motion.div>
            );
          })}

          {filtered.length === 0 && (
            <p className="text-white/40 text-center py-8">No devices match your filter</p>
          )}
        </div>
      </motion.div>
    </div>
  );
}