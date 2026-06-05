import { useRef } from 'react';
import { motion } from 'framer-motion';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Line, Stars } from '@react-three/drei';
import * as THREE from 'three';

function Globe() {
    const meshRef = useRef();
    useFrame(({ clock }) => {
        if (meshRef.current) {
            meshRef.current.rotation.y = clock.getElapsedTime() * 0.1;
        }
    });

    const points = [];
    for (let i = 0; i < 40; i++) {
        const phi = Math.acos(-1 + (2 * i) / 40);
        const theta = Math.sqrt(40 * Math.PI) * phi;
        points.push([
            Math.cos(theta) * Math.sin(phi),
            Math.sin(theta) * Math.sin(phi),
            Math.cos(phi),
        ]);
    }

    return (
        <group ref={meshRef}>
            <Sphere args={[1, 32, 32]}>
                <meshStandardMaterial
                    color="#00ffc8"
                    wireframe
                    opacity={0.3}
                    transparent
                />
            </Sphere>
            <Sphere args={[1.01, 64, 64]}>
                <meshBasicMaterial color="#00aaff" opacity={0.05} transparent />
            </Sphere>

            {points.map((p, i) => (
                <Sphere key={i} args={[0.02]} position={p}>
                    <meshBasicMaterial color={i % 5 === 0 ? '#ff0055' : '#00ffc8'} />
                </Sphere>
            ))}
        </group>
    );
}

export default function Globe3D() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-xl p-6 h-[400px] relative overflow-hidden"
        >
            <h3 className="text-xl font-semibold mb-2 flex items-center gap-2">
                Global Threat Map
            </h3>
            <p className="text-xs text-white/50 mb-4">Real-time device geolocation</p>

            <div className="h-[300px]">
                <Canvas camera={{ position: [0, 0, 3] }}>
                    <ambientLight intensity={0.3} />
                    <pointLight position={[5, 5, 5]} color="#00ffc8" intensity={2} />
                    <Stars radius={50} depth={50} count={500} factor={2} fade />
                    <Globe />
                    <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
                </Canvas>
            </div>
        </motion.div>
    );
}