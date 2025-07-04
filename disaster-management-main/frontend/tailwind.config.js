// tailwind.config.js
module.exports = {
  content: [
    "./index.html",                // for Vite's entry point
    "./src/**/*.{js,jsx,ts,tsx}",   // for React components and pages
  ],
  theme: {
    extend: {
      // colors: {
      //   navy: '#001f3d',        // Navy Blue
      //   gray: '#6c757d',        // Gray
      //   teal: '#3a9a8f',        // Teal
      //   white: '#ffffff',       // White
      //   lightGray: '#d3d3d3',   // Light Gray
      // },
      colors: {
        navy: {
          DEFAULT: '#1E3A8A',
          dark: '#14274E',
        },
        teal: {
          DEFAULT: '#14B8A6',
          light: '#E0F2F1',
        },
        gray: {
          light: '#F3F4F6',
          DEFAULT: '#6B7280',
        },
        yellow: '#FACC15',
        red: '#EF4444',
        green: '#22C55E',
        orange: '#FFA500'
      },
      animation: {
        marquee: 'marquee 30s linear infinite',
      },
      keyframes: {
        marquee: {
          '0%': { transform: 'translateX(100%)' },
          '100%': { transform: 'translateX(-100%)' },
        },
      },
      fontFamily: {
        sans: ['Arial', 'sans-serif'],
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
  ],
}
