import { motion } from 'framer-motion';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, Float, Stars } from '@react-three/drei';
import { Shield, Cpu, Radio, Eye } from 'lucide-react';

function HorusOrb() {
    return (
        <Float speed={2} rotationIntensity={1} floatIntensity={2}>
            <Sphere args={[1, 64, 64]}>
                <meshStandardMaterial
                    color="#00ffc8"
                    emissive="#00ffc8"
                    emissiveIntensity={0.5}
                    wireframe
                />
            </Sphere>
            <Sphere args={[1.2, 32, 32]}>
                <meshStandardMaterial
                    color="#00aaff"
                    transparent
                    opacity={0.1}
                />
            </Sphere>
        </Float>
    );
}

export default function HeroSection() {
    return (
        <section className="relative min-h-[70vh] flex items-center justify-between gap-12">
            <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                className="flex-1 max-w-2xl"
            >
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-horus-cyan/30 bg-horus-cyan/10 mb-6">
                    <div className="w-2 h-2 rounded-full bg-horus-green animate-pulse" />
                    <span className="text-xs mono text-horus-cyan">SYSTEM ACTIVE</span>
                </div>

                <h1 className="text-6xl font-bold mb-6 leading-tight">
                    <span className="gradient-text">Detect.</span>{' '}
                    <span className="text-white">Track.</span>{' '}
                    <span className="gradient-text">Neutralize.</span>
                </h1>

                <p className="text-xl text-white/70 mb-8 leading-relaxed">
                    An intelligent anti-drone detection system combining microwave radar, PIR motion, barometric pressure, WiFi sniffing, and AI-powered camera vision into one unified threat response platform.
                </p>

                <div className="grid grid-cols-2 gap-4 mb-8">
                    {[
                        { icon: Radio, label: '4× RCWL Radar', value: '360° coverage' },
                        { icon: Eye, label: 'AI Camera', value: 'Real-time tracking' },
                        { icon: Cpu, label: 'ESP32 Brain', value: 'Dual-core 240MHz' },
                        { icon: Shield, label: 'RM6W Relay', value: '6-channel response' },
                    ].map((item, i) => (
                        <motion.div
                            key={item.label}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.1 + 0.3 }}
                            className="glass rounded-lg p-4 flex items-center gap-3"
                        >
                            <item.icon className="w-6 h-6 text-horus-cyan" />
                            <div>
                                <p className="text-sm text-white/60">{item.label}</p>
                                <p className="text-sm mono text-horus-cyan">{item.value}</p>
                            </div>
                        </motion.div>
                    ))}
                </div>

                <div className="flex gap-4">
                    <button className="px-6 py-3 bg-horus-cyan text-black font-medium rounded-lg hover:bg-horus-cyan/80 transition-all">
                        View Live Feed
                    </button>
                    <button className="px-6 py-3 border border-horus-cyan/50 text-horus-cyan rounded-lg hover:bg-horus-cyan/10 transition-all">
                        System Status
                    </button>
                </div>
            </motion.div>

            <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 1, delay: 0.3 }}
                className="flex-1 h-[500px] hidden lg:block"
            >
                <Canvas camera={{ position: [0, 0, 4] }}>
                    <ambientLight intensity={0.5} />
                    <pointLight position={[10, 10, 10]} color="#00ffc8" intensity={2} />
                    <pointLight position={[-10, -10, -10]} color="#ff00aa" intensity={1} />
                    <Stars radius={50} depth={50} count={1000} factor={4} fade speed={1} />
                    <HorusOrb />
                    <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
                </Canvas>
            </motion.div>
        </section>
    );
}