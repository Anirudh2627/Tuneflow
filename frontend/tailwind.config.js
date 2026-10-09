/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          950: '#09090f',
          900: '#0f1018',
          850: '#15161f',
          800: '#1c1e2a',
          700: '#262938',
        },
        accent: {
          DEFAULT: '#8b5cf6',
          soft: '#a78bfa',
          dim: '#6d28d9',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
