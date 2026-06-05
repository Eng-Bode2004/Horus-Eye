import axios from 'axios';
import { io } from 'socket.io-client';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001';
const WS_URL = import.meta.env.VITE_WS_URL || 'http://localhost:3001';

export const horusAPI = {
  getDevices: () => axios.get(`${BACKEND_URL}/api/devices`),
  getSensors: () => axios.get(`${BACKEND_URL}/api/sensors`),
  getStatus: () => axios.get(`${BACKEND_URL}/api/status`),
  getDrones: () => axios.get(`${BACKEND_URL}/api/drones`),
  getAlerts: (limit) => axios.get(`${BACKEND_URL}/api/alerts?limit=${limit}`),
  getStats: () => axios.get(`${BACKEND_URL}/api/stats`),
  getAiDetections: () => axios.get(`${BACKEND_URL}/api/ai/detections`),
  exportCSV: () => `${BACKEND_URL}/api/export/csv`,
};

export function connectBackend(onDevices, onSensors) {
  const socket = io(WS_URL, {
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: Infinity,
  });

  socket.on('devices_update', (data) => {
    if (onDevices) onDevices(data);
  });

  socket.on('sensors_update', (data) => {
    if (onSensors) onSensors(data);
  });

  socket.on('drone_alert', (data) => {
    console.log('[WS] Drone alert:', data);
    window.__HORUS_DRONE_ALERTS__ = (window.__HORUS_DRONE_ALERTS__ || []).concat(data.drones);
    // Dispatch custom event for any component to listen
    window.dispatchEvent(new CustomEvent('droneAlert', { detail: data }));
  });

  socket.on('drone_status', (data) => {
    console.log('[WS] Drone status:', data.count, 'drones');
    window.__HORUS_DRONE_STATUS__ = data;
    window.dispatchEvent(new CustomEvent('droneStatus', { detail: data }));
  });

  socket.on('ai_detection', (data) => {
    console.log('[WS] AI detection:', data.drone_count, 'drones,', data.total_detections, 'total');
    window.__HORUS_AI_DETECTION__ = data;
    window.dispatchEvent(new CustomEvent('aiDetection', { detail: data }));
  });



  socket.on('connect', () => console.log('[WS] Connected to backend'));
  socket.on('disconnect', () => console.log('[WS] Disconnected from backend'));
  socket.on('connect_error', (err) => console.error('[WS] Connection error:', err.message));

  return socket;
}

export function onDroneAlert(callback) {
  const handler = (e) => callback(e.detail);
  window.addEventListener('droneAlert', handler);
  return () => window.removeEventListener('droneAlert', handler);
}

export function onDroneStatus(callback) {
  const handler = (e) => callback(e.detail);
  window.addEventListener('droneStatus', handler);
  return () => window.removeEventListener('droneStatus', handler);
}

export function onAiDetection(callback) {
  const handler = (e) => callback(e.detail);
  window.addEventListener('aiDetection', handler);
  return () => window.removeEventListener('aiDetection', handler);
}

