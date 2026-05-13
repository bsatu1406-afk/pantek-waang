/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "hsl(222 47% 7%)",
        foreground: "hsl(210 40% 98%)",
        muted: "hsl(217 33% 17%)",
        "muted-foreground": "hsl(215 20% 65%)",
        border: "hsl(215 28% 17%)",
        primary: "hsl(217 91% 60%)",
        "primary-foreground": "hsl(0 0% 100%)",
        accent: "hsl(217 33% 17%)",
        "accent-foreground": "hsl(210 40% 98%)",
        destructive: "hsl(0 70% 45%)",
        "destructive-foreground": "hsl(0 0% 100%)",
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
