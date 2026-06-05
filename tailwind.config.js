export default {
    content: ["./index.html", "./src/**/*.{js,jsx}"],
    theme: {
        extend: {
            colors: {
                horus: {
                    cyan: '#00ffc8',
                    blue: '#00aaff',
                    red: '#ff0055',
                    amber: '#ffaa00',
                    green: '#00ff88',
                    dark: '#0a0e1a',
                    darker: '#050810',
                }
            },
            fontFamily: {
                mono: ['JetBrains Mono', 'monospace'],
            },
        },
    },
    plugins: [],
};