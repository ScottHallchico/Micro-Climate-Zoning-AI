/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B0F19',
        surface: '#1A233A',
        surfaceHover: '#2A3655',
        primary: '#3B82F6',
        primaryHover: '#60A5FA',
        accent: '#10B981',
      }
    },
  },
  plugins: [],
}
