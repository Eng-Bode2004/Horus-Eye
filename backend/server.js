/**
 * HORUS EYE — Backend Bridge Server v2.0
 * Connects ESP32 → AI Classifier → Frontend via WebSocket
 * Features: Adaptive polling, drone alerts, CSV export, stats
 */

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { classifyDevices, getStats, exportCSV } = require('./classifier');

const ESP32_IP = process.env.ESP32_IP || '10.51.69.152';
const ESP32_HTTP = `http://${ESP32_IP}`;
const PORT = process.env.PORT || 3001;
const BASE_POLL_INTERVAL = 2000;
const IDLE_POLL_INTERVAL = 10000;

// ==================== APP SETUP ====================

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: ['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173'], methods: ['GET', 'POST'] }
});

app.use(express.json());

// ==================== AI DETECTION CACHE ====================

let cachedAiDetection = null;

// ==================== STATE ====================

let cachedDevices = [];
let cachedSensors = {};
let lastFetchTime = 0;
let droneAlertHistory = [];   // Track drone detection events
let lastKnownDeviceCount = 0;
let consecutiveFailures = 0;
const MAX_FAILURES_BEFORE_MOCK = 3;
let useMockData = false;
let mockDeviceCount = 0;
const MOCK_MACS = [
  'AA:BB:CC:DD:EE:01', 'AA:BB:CC:DD:EE:02', 'AA:BB:CC:DD:EE:03',
  'AA:BB:CC:DD:EE:04', 'AA:BB:CC:DD:EE:05', 'AA:BB:CC:DD:EE:06',
  '12:34:56:78:9A:BC', 'FE:DC:BA:98:76:54', '00:11:22:33:44:55',
  'A0:B1:C2:D3:E4:F5'
];

function generateMockDevices() {
  const types = ['phone', 'laptop', 'router', 'iot', 'drone', 'other'];
  const methods = ['oui_db', 'local_db', 'external_api', 'random_mac_algo', 'feature_analysis'];
  const vendors = ['Apple', 'Samsung', 'Intel', 'Broadcom', 'DJI', 'TP-Link', 'Huawei', 'Unknown'];

  // Vary count to simulate real traffic
  const count = 5 + Math.floor(Math.abs(Math.sin(Date.now() / 10000)) * 10);
  mockDeviceCount = count;
  const devices = [];

  for (let i = 0; i < count; i++) {
    const rssi = -35 - Math.random() * 55;
    const chan = 1 + Math.floor(Math.random() * 13);
    const typeIdx = i === 0 ? 4 : Math.floor(Math.random() * (types.length - 1));
    const mac = MOCK_MACS[i % MOCK_MACS.length];

    devices.push({
      mac,
      rssi: Math.round(rssi * 10) / 10,
      avg_rssi: Math.round((rssi + Math.random() * 10 - 5) * 10) / 10,
      std_rssi: Math.round(Math.random() * 5 * 10) / 10,
      min_rssi: Math.round((rssi - Math.random() * 10) * 10) / 10,
      max_rssi: Math.round((rssi + Math.random() * 10) * 10) / 10,
      rssi_variation: Math.round(Math.random() * 15 * 10) / 10,
      channel: chan,
      channels_seen: 1 + Math.floor(Math.random() * 3),
      packets: 10 + Math.floor(Math.random() * 200),
      packet_growth: 1 + Math.floor(Math.random() * 20),
      is_random_mac: Math.random() > 0.7 ? 1 : 0,
      type: types[typeIdx],
      vendor: vendors[Math.floor(Math.random() * vendors.length)],
      confidence: Math.round((0.5 + Math.random() * 0.5) * 100) / 100,
      method: methods[Math.floor(Math.random() * methods.length)],
      estimatedDist: Math.round((0.3 + Math.random() * 50) * 10) / 10
    });
  }

  return devices;
}

function generateMockSensors() {
  const modes = ['NORMAL', 'HEAT_WAVE', 'STORM', 'THREAT_OVERHEAD', 'COLD'];
  const zones = ['NONE', 'OUTER', 'INNER', 'PROXIMITY'];
  const temp = 18 + Math.sin(Date.now() / 5000) * 8;
  const pressure = 1013 + Math.sin(Date.now() / 15000) * 10;
  const alt = 44330 * (1 - Math.pow(pressure / 1013.25, 1 / 5.255));

  return {
    temperature: Math.round(temp * 10) / 10,
    pir_active: Math.random() > 0.8,
    radar_sky: Math.random() > 0.85,
    radar_left: Math.random() > 0.85,
    radar_right: Math.random() > 0.85,
    radar_back: Math.random() > 0.85,
    alert_active: Math.random() > 0.9,
    device_count: mockDeviceCount,
    mode: modes[Math.floor(Math.random() * modes.length)],
    zone: zones[Math.floor(Math.random() * zones.length)],
    pressure: Math.round(pressure * 100) / 100,
    altitude: Math.round(alt * 10) / 10,
    vertical_velocity: Math.round((Math.random() - 0.5) * 3 * 10) / 10
  };
}

// ==================== ESP32 DATA FETCHER ====================

async function fetchFromESP32(endpoint) {
  try {
    const url = `${ESP32_HTTP}${endpoint}`;
    const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) {
      console.log(`[ESP32] ${endpoint} → ${resp.status} ${resp.statusText}`);
      return null;
    }
    const data = await resp.json();
    console.log(`[ESP32] ${endpoint} → OK (${Array.isArray(data) ? data.length + ' items' : 'object'})`);
    return data;
  } catch (err) {
    console.log(`[ESP32] ${endpoint} → FAILED: ${err.message || err}`);
    return null;
  }
}

// ==================== ADAPTIVE POLLING ====================

function getPollInterval(devices) {
  const activeCount = devices.filter(d => d.type === 'drone').length;
  if (activeCount > 0) return Math.max(1000, BASE_POLL_INTERVAL / 2); // Faster when drones detected
  if (devices.length === 0) return IDLE_POLL_INTERVAL;             // Slow when empty
  if (devices.length === lastKnownDeviceCount) return BASE_POLL_INTERVAL * 1.5; // Slower when stable
  return BASE_POLL_INTERVAL;
}

// ==================== DRONE ALERT ENGINE ====================

function detectNewDrones(devices) {
  const currentDrones = devices.filter(d => d.type === 'drone');
  const previousDroneMacs = cachedDevices.filter(d => d.type === 'drone').map(d => d.mac);
  const newDrones = currentDrones.filter(d => !previousDroneMacs.includes(d.mac));

  newDrones.forEach(d => {
    const alert = {
      mac: d.mac,
      vendor: d.vendor,
      rssi: d.rssi,
      channel: d.channel,
      confidence: d.confidence,
      method: d.method,
      timestamp: Date.now(),
      estimatedDist: estimateDistance(d.rssi)
    };
    droneAlertHistory.unshift(alert);
    if (droneAlertHistory.length > 50) droneAlertHistory.pop();
    console.log(`🚨 DRONE ALERT: ${d.mac} (${d.vendor}) — RSSI: ${d.rssi} dBm — Method: ${d.method}`);
  });

  return newDrones;
}

function estimateDistance(rssi) {
  if (!rssi || rssi >= -10) return 0.3;
  if (rssi < -100) return 150;
  const RSSI_AT_1M = -43;
  const PATH_LOSS_N = 2.8;
  const d = Math.pow(10, (RSSI_AT_1M - rssi) / (10 * PATH_LOSS_N));
  return Math.max(0.3, Math.min(150, d));
}

// ==================== MAIN POLL LOOP ====================

let pollTimer = null;

async function pollESP32() {
  try {
    // Fetch devices
    const devices = await fetchFromESP32('/api/devices');
    if (devices && Array.isArray(devices)) {
      consecutiveFailures = 0;
      useMockData = false;
      const classified = await classifyDevices(devices);

      // Detect new drones and emit alerts
      const newDrones = detectNewDrones(classified);
      if (newDrones.length > 0) {
        io.emit('drone_alert', {
          drones: newDrones,
          allDrones: classified.filter(d => d.type === 'drone'),
          alertHistory: droneAlertHistory.slice(0, 20)
        });
      }

      // Emit periodic drone status update
      const droneCount = classified.filter(d => d.type === 'drone').length;
      if (droneCount > 0 || classified.length !== lastKnownDeviceCount) {
        io.emit('drone_status', {
          count: droneCount,
          devices: classified.filter(d => d.type === 'drone'),
          history: droneAlertHistory.slice(0, 10)
        });
      }

      cachedDevices = classified;
      lastKnownDeviceCount = classified.length;
      io.emit('devices_update', { devices: classified, mock: false });
    } else {
      // ESP32 fetch failed — count consecutive failures
      consecutiveFailures++;
      if (consecutiveFailures >= MAX_FAILURES_BEFORE_MOCK && !useMockData) {
        useMockData = true;
        console.log(`[MOCK] ESP32 unreachable for ${consecutiveFailures} polls — switching to simulated data`);
      }
    }

    // Fetch sensors
    const sensors = await fetchFromESP32('/api/sensors');
    if (sensors) {
      cachedSensors = { ...sensors, mock: false };
      io.emit('sensors_update', cachedSensors);
    }

    lastFetchTime = Date.now();
  } catch (err) {
    console.error('[POLL ERROR]', err.message);
    consecutiveFailures++;
    if (consecutiveFailures >= MAX_FAILURES_BEFORE_MOCK && !useMockData) {
      useMockData = true;
      console.log(`[MOCK] ESP32 unreachable — switching to simulated data`);
    }
  }

  // If in mock mode, generate and emit synthetic data
  if (useMockData) {
    const mockDevices = generateMockDevices();
    const mockSensors = generateMockSensors();

    cachedDevices = mockDevices;
    cachedSensors = mockSensors;
    lastKnownDeviceCount = mockDevices.length;

    io.emit('devices_update', { devices: mockDevices, mock: true });
    io.emit('sensors_update', { ...mockSensors, mock: true });
  }

  // Schedule next poll with adaptive interval
  const interval = useMockData ? BASE_POLL_INTERVAL : getPollInterval(cachedDevices);
  pollTimer = setTimeout(pollESP32, interval);
}

// ==================== REST API ====================

app.get('/', (req, res) => {
  const stats = getStats();
  res.json({
    name: 'HORUS EYE Backend Bridge',
    version: '2.0.0',
    status: 'running',
    esp32: ESP32_IP,
    frontend: 'http://localhost:3000',
    uptime: process.uptime(),
    lastFetch: lastFetchTime,
    devices: cachedDevices.length,
    droneAlerts: droneAlertHistory.length,
    classifierStats: stats,
    endpoints: {
      devices: '/api/devices',
      sensors: '/api/sensors',
      status: '/api/status',
      drones: '/api/drones',
      alerts: '/api/alerts',
      export: '/api/export/csv',
      stats: '/api/stats',
      aiDetection: '/api/ai/detections'
    }
  });
});

app.get('/api/devices', (req, res) => {
  res.json(cachedDevices);
});

app.get('/api/sensors', (req, res) => {
  res.json(cachedSensors);
});

app.get('/api/status', (req, res) => {
  const stats = getStats();
  res.json({
    online: cachedDevices.length > 0 || Date.now() - lastFetchTime < 15000,
    devicesDetected: cachedDevices.length,
    droneCount: cachedDevices.filter(d => d.type === 'drone').length,
    droneAlerts: droneAlertHistory.length,
    esp32Ip: ESP32_IP,
    lastUpdate: lastFetchTime,
    uptime: process.uptime(),
    classifier: stats
  });
});

app.get('/api/drones', (req, res) => {
  const drones = cachedDevices.filter(d => d.type === 'drone');
  res.json({
    count: drones.length,
    drones,
    history: droneAlertHistory.slice(0, 20)
  });
});

app.get('/api/ai/detections', (req, res) => {
  res.json(cachedAiDetection || { detections: [], drone_count: 0, timestamp: null, source: null });
});

app.get('/api/alerts', (req, res) => {
  res.json({
    count: droneAlertHistory.length,
    alerts: droneAlertHistory.slice(0, parseInt(req.query.limit) || 20)
  });
});

app.get('/api/export/csv', (req, res) => {
  const csv = exportCSV(cachedDevices);
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename=horus-devices-${Date.now()}.csv`);
  res.send(csv);
});

app.get('/api/stats', (req, res) => {
  res.json(getStats());
});

// ==================== SOCKET.IO (Frontend) ====================

io.on('connection', (socket) => {
  console.log(`[IO] Client connected: ${socket.id}`);

  // ===== AI Camera detection listener =====
  socket.on('ai_detection', (data) => {
    cachedAiDetection = { ...data, timestamp: Date.now() };
    socket.broadcast.emit('ai_detection', cachedAiDetection);
  });

  // ===== Frontend connections =====
  // Send cached data immediately
  socket.emit('devices_update', { devices: cachedDevices, mock: useMockData });
  socket.emit('sensors_update', { ...cachedSensors, mock: useMockData });
  socket.emit('drone_status', {
    count: cachedDevices.filter(d => d.type === 'drone').length,
    devices: cachedDevices.filter(d => d.type === 'drone'),
    history: droneAlertHistory.slice(0, 10)
  });
  if (cachedAiDetection) {
    socket.emit('ai_detection', cachedAiDetection);
  }

  socket.on('disconnect', () => {
    console.log(`[IO] Client disconnected: ${socket.id}`);
  });
});

// ==================== STARTUP ====================

async function start() {
  server.listen(PORT, () => {
    console.log(`\n╔══════════════════════════════════════════════╗`);
    console.log(`║     HORUS EYE — Backend Bridge v2.0          ║`);
    console.log(`╠══════════════════════════════════════════════╣`);
    console.log(`║  Frontend  → http://localhost:${PORT}          ║`);
    console.log(`║  ESP32     → ${ESP32_IP}                    ║`);
    console.log(`║  Adaptive  → ${BASE_POLL_INTERVAL}ms base            ║`);
    console.log(`╚══════════════════════════════════════════════╝\n`);
  });

  // Initial fetch
  await pollESP32();
}

start().catch(console.error);