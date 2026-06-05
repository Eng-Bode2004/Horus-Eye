/**
 * HORUS EYE — Device Classifier v7
 * Merges: Local OUI DB + External API + Feature Analysis
 * Rule: Feature analysis NEVER runs on random MACs
 * Rule: External API vendor names are cross-referenced with OUI DB for type
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

// ============================================================
// LOAD EXPANDED LOCAL OUI DATABASE (867 entries)
// ============================================================
let vendorDb = {};
try {
  vendorDb = JSON.parse(fs.readFileSync(path.join(__dirname, 'oui', 'vendors.json'), 'utf8'));
  const droneEntries = Object.values(vendorDb).filter(v => v.type === 'drone');
  console.log(`[CLASSIFIER] Loaded ${Object.keys(vendorDb).length} OUI entries, ${droneEntries.length} drone families`);
} catch (err) {
  console.log('[CLASSIFIER] Warning: Could not load vendors.json:', err.message);
}

// Build vendor-name → type lookup from local DB (for cross-referencing API results)
const VENDOR_TYPE_MAP = new Map();
for (const [oui, entry] of Object.entries(vendorDb)) {
  const name = (entry.vendor || '').split('/')[0].trim().toLowerCase();
  if (name && !name.includes('unknown') && !name.includes('private')) {
    if (!VENDOR_TYPE_MAP.has(name) || entry.type === 'phone' || entry.type === 'router') {
      VENDOR_TYPE_MAP.set(name, entry.type);
    }
  }
}
// Also add well-known vendor → type mappings not dependent on OUI matches
const FALLBACK_VENDOR_TYPES = {
  'samsung': 'phone', 'apple': 'phone', 'huawei': 'phone', 'xiaomi': 'phone',
  'google': 'phone', 'oneplus': 'phone', 'lg': 'phone', 'sony': 'phone',
  'nokia': 'phone', 'htc': 'phone', 'motorola': 'phone', 'oppo': 'phone',
  'vivo': 'phone', 'realme': 'phone', 'honor': 'phone', 'zte': 'phone',
  'oneplus': 'phone', 'pixel': 'phone',
  'intel': 'laptop', 'laptop': 'laptop', 'notebook': 'laptop',
  'cisco': 'router', 'tplink': 'router', 'tp-link': 'router', 'netgear': 'router',
  'linksys': 'router', 'dlink': 'router', 'd-link': 'router', 'asus': 'router',
  'ubiquiti': 'router', 'mikrotik': 'router', 'ruckus': 'router', 'aruba': 'router',
  'espressif': 'iot', 'raspberry': 'iot', 'arduino': 'iot', 'particle': 'iot',
  'microchip': 'iot', 'silicon': 'iot', 'nordic': 'iot',
  'microsoft': 'laptop', 'lenovo': 'laptop', 'dell': 'laptop', 'hp ': 'laptop',
  'hewlett': 'laptop', 'packard': 'laptop', 'acer': 'laptop',
};

// ============================================================
// KNOWN DRONE OUIs
// ============================================================
const DRONE_OUI = new Set([
  '60:60:1F','48:1C:B9','F8:1D:78','A0:14:3D','A6:B0:1A',
  '00:12:1C','00:26:7E','04:A0:2B','34:A8:45','00:24:9B',
  '00:0A:3A','00:06:1A','00:0B:0C','38:1D:14'
]);

const DRONE_FAMILIES = new Set(['dji', 'parrot', 'skydio', '3dr', 'autel']);
const RANDOM_OUI_6 = new Set([   // These never have useful OUI info
  '52:C6:7B','62:6E:2B'
]);

function getVendor(mac) {
  return vendorDb[mac.substring(0,8).toUpperCase()]?.vendor || null;
}

function isRandomMac(mac) {
  return (parseInt(mac.substring(0,2),16) & 0x02) ? 1 : 0;
}

function inferTypeFromVendor(vendorName) {
  if (!vendorName) return null;
  const lower = vendorName.toLowerCase();

  // Check VENDOR_TYPE_MAP built from local DB
  for (const [key, type] of VENDOR_TYPE_MAP) {
    if (lower.includes(key)) return type;
  }

  // Check fallback map
  for (const [key, type] of Object.entries(FALLBACK_VENDOR_TYPES)) {
    if (lower.includes(key)) return type;
  }

  return null;
}

// ============================================================
// EXTERNAL API — cached + rate-limited
// ============================================================
const ouiCache = new Map();
const seen404 = new Set();
const seen429 = new Map();
const BLOCKED_MS = 120000;
const MAX_DAILY_API_CALLS = 100;
const COOLDOWN_MS = 1200;
let lastApiMs = 0;
let apiCallsToday = 0;
let dailyResetAt = Date.now() + 86400000;

function checkDailyBudget() {
  if (Date.now() > dailyResetAt) {
    apiCallsToday = 0;
    dailyResetAt = Date.now() + 86400000;
    console.log('[AGENT] Daily API budget reset');
  }
  return apiCallsToday < MAX_DAILY_API_CALLS;
}

async function queryExternalAgent(mac) {
  const oui = mac.substring(0,8).toUpperCase();
  if (ouiCache.has(oui)) return ouiCache.get(oui);
  if (seen404.has(oui)) return null;

  if (seen429.has(oui)) {
    const backoffLeft = BLOCKED_MS - (Date.now() - seen429.get(oui));
    if (backoffLeft > 0) return null;
    seen429.delete(oui);
  }

  if (!checkDailyBudget()) return null;

  const elapsed = Date.now() - lastApiMs;
  if (elapsed < COOLDOWN_MS) await new Promise(r => setTimeout(r, COOLDOWN_MS - elapsed));

  try {
    const vendorName = await fetchVendorName(oui);
    lastApiMs = Date.now();
    apiCallsToday++;
    if (vendorName) {
      ouiCache.set(oui, vendorName);
      return vendorName;
    }
    return null;
  } catch (err) {
    lastApiMs = Date.now();
    if (err.message.includes('404')) { seen404.add(oui); return null; }
    if (err.message.includes('429')) { seen429.set(oui, Date.now()); return null; }
    return null;
  }
}

function fetchVendorName(oui) {
  return new Promise((resolve, reject) => {
    const req = https.get('https://api.macvendors.com/'+oui, {timeout:5000}, (res) => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        if (res.statusCode===200 && body.trim()) resolve(body.trim());
        else reject(new Error('HTTP '+res.statusCode));
      });
    });
    req.on('error', reject);
    req.on('timeout', ()=>{req.destroy(); reject(new Error('timeout'))});
    req.end();
  });
}

// ============================================================
// MAIN CLASSIFIER — merged approach
// ============================================================
async function classifyDevice(device){
  const mac=(device.mac||'').toUpperCase();
  const oui=mac.substring(0,8);
  const randomMac = device.is_random_mac !== undefined ? device.is_random_mac : isRandomMac(mac);
  const packetCount = device.packets || device.max_packets || 0;
  const channelCount = device.channels_seen || 1;
  const rssiVariation = device.rssi_variation || (device.max_rssi - device.min_rssi) || 0;

  const features = {
    avg_rssi: device.avg_rssi || device.rssi || -80,
    rssi_variation: rssiVariation,
    channels_seen: channelCount,
    max_packets: packetCount,
    packet_growth: device.packet_growth || 0,
    is_random_mac: randomMac,
  };

  // ---- STEP 1: Check hardcoded drone OUI ----
  if (DRONE_OUI.has(oui)) {
    return { mac, vendor: getVendor(mac) || 'Drone Manufacturer', type: 'drone', confidence: 0.95, method: 'drone_oui', features };
  }

  // ---- STEP 2: Check local OUI database ----
  const localEntry = vendorDb[oui];
  if (localEntry) {
    if (localEntry.type === 'drone' || DRONE_FAMILIES.has(localEntry.family)) {
      return { mac, vendor: localEntry.vendor, type: 'drone', confidence: 0.93, method: 'local_oui_db', features };
    }
    return { mac, vendor: localEntry.vendor, type: localEntry.type, confidence: 0.88, method: 'local_oui_db', features };
  }

  // ---- STEP 3: Try external API ----
  const apiVendor = await queryExternalAgent(mac);
  if (apiVendor) {
    // Check if this vendor name matches a known drone manufacturer
    const droneKeywords = ['dji','parrot','autel','hubsan','3dr','yuneec','syma','blade','walkera','skydio','freefly'];
    const isDrone = droneKeywords.some(k => apiVendor.toLowerCase().includes(k));
    if (isDrone) {
      return { mac, vendor: apiVendor, type: 'drone', confidence: 0.92, method: 'api_identified', features };
    }

    // Infer device type from vendor name
    const inferredType = inferTypeFromVendor(apiVendor);
    if (inferredType) {
      return { mac, vendor: apiVendor, type: inferredType, confidence: 0.85, method: 'api_identified', features };
    }

    // Vendor known but type unknown — still show it as "other" with the vendor name
    return { mac, vendor: apiVendor, type: 'other', confidence: 0.65, method: 'api_identified', features };
  }

  // ---- STEP 4: Handle random MACs (privacy addresses) ----
  if (randomMac) {
    // Randomized MACs are almost always modern phones/PCs with MAC randomization
    // High packet count + high RSSI variation = active phone
    // Low packet count = briefly scanned

    const isHighActivity = packetCount > 100 || channelCount > 5;
    const isClose = (device.rssi || -80) > -60;
    let type = 'phone';
    let desc = 'Randomized MAC (Modern Phone)';

    if (packetCount > 1000 && channelCount > 5) {
      type = 'phone';
      desc = 'Active Phone (Randomized MAC)';
    } else if (packetCount > 500) {
      type = 'phone';
      desc = 'Phone (Randomized MAC)';
    } else if (packetCount < 10) {
      // Very brief appearance — could be any device with randomization
      desc = 'Randomize MAC (Modern Devices)';
    }

    const conf = Math.min(0.85, 0.40 + (packetCount / 2000) * 0.3 + (channelCount / 13) * 0.15);
    return { mac, vendor: desc, type, confidence: Math.round(conf * 100) / 100, method: 'randomized_mac', features };
  }

  // ---- STEP 5: Feature analysis (non-random MACs only) ----
  // For non-random MACs with real OUI prefixes, we can try statistical classification
  if (packetCount > 0) {
    const result = featureClassify(features);
    if (result.type !== 'other' || result.confidence > 0.35) {
      return { mac, vendor: null, type: result.type, confidence: result.confidence, method: 'feature_analysis', features };
    }
  }

  // Absolute fallback
  return { mac, vendor: null, type: 'other', confidence: 0.25, method: 'insufficient_data', features };
}

// ============================================================
// FEATURE ANALYSIS (NEVER drone, non-random MACs only)
// ============================================================
function logG(x,m,s){ if(s<0.01)return 0; return -0.5*Math.log(2*Math.PI)-Math.log(s)-((x-m)**2)/(2*s*s); }

function featureClassify(f) {
  const P = {
    phone:{r:-55,v:12,c:1.8,g:150,p:250,n:0.5},
    router:{r:-75,v:4,c:1.1,g:500,p:600,n:0},
    laptop:{r:-60,v:8,c:1.3,g:150,p:200,n:0.3},
    iot:{r:-80,v:3,c:1,g:5,p:10,n:0.05},
    other:{r:-85,v:8,c:1.5,g:30,p:40,n:0.3},
  };
  const S={r:15,v:10,c:1.5,g:200,p:300,n:0.2};
  const W={r:1,v:2.5,c:3,g:1.5,p:0.5,n:0.2};
  const sc={};
  for(const[cls,pr] of Object.entries(P)){
    let s=0;
    s+=logG(f.avg_rssi,pr.r,S.r)*W.r;
    s+=logG(f.rssi_variation,pr.v,S.v)*W.v;
    s+=logG(f.channels_seen,pr.c,S.c)*W.c;
    s+=logG(f.packet_growth,pr.g,S.g)*W.g;
    s+=logG(f.max_packets,pr.p,S.p)*W.p;
    const rl=f.is_random_mac===1?pr.n:(1-pr.n);
    s+=Math.log(Math.max(rl,0.001))*W.n;
    sc[cls]=s;
  }
  let best='other',bs=-Infinity;
  for(const[cls,s]of Object.entries(sc)){if(s>bs){bs=s;best=cls;}}
  const so=Object.entries(sc).sort((a,b)=>b[1]-a[1]);
  const conf=Math.max(0.2,Math.min(0.75,0.25+(so[0][1]-so[1][1])*0.08));
  return {type:best, confidence:Math.round(conf*100)/100};
}

// ============================================================
// EXPORTS
// ============================================================
async function classifyDevices(devices){
  return Promise.all(devices.map(async d=>{
    const r=await classifyDevice(d);
    return {
      mac: r.mac,
      vendor: r.vendor || 'Unknown',
      type: r.type,
      rssi: d.rssi || d.avg_rssi || 0,
      channel: d.channel || 1,
      packets: d.packets || d.max_packets || 0,
      confidence: r.confidence,
      method: r.method,
      features: r.features
    };
  }));
}

function getStats() {
  return {
    cacheSize: ouiCache.size,
    seen404: seen404.size,
    seen429: seen429.size,
    apiCallsToday,
    dailyBudget: MAX_DAILY_API_CALLS,
    localDbSize: Object.keys(vendorDb).length,
    droneFamilies: Object.values(vendorDb).filter(v => v.type === 'drone').length
  };
}

function exportCSV(devices) {
  const header = 'mac,vendor,type,confidence,method,rssi,channel,packets';
  const rows = devices.map(d => [
    d.mac, `"${d.vendor||''}"`, d.type, d.confidence, d.method, d.rssi, d.channel, d.packets
  ].join(','));
  return [header, ...rows].join('\n');
}

module.exports = { classifyDevice, classifyDevices, getVendor, getStats, exportCSV };