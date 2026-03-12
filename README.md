# Mini-LIDAR Air Quality Sensor — Research Website

Research-grade project website for the **Laser-Based Low-Cost Aerosol/PM Sensor (Mini-LIDAR)** system.

B.Tech Major Project · Department of Physics · NIT Hamirpur · 2025–26

## Deploy to Vercel

### Option 1: Vercel CLI
```bash
npm i -g vercel
vercel --prod
```

### Option 2: GitHub → Vercel
1. Push this repo to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import the repository
4. Vercel auto-detects `vercel.json` → deploys `public/` folder
5. Click **Deploy**

### Option 3: Drag & Drop
1. Go to [vercel.com/new](https://vercel.com/new)
2. Drag the `public/` folder onto the page

## Local Preview
```bash
npx serve public
# Opens at http://localhost:3000
```

## Structure
```
mini-lidar-web/
├── vercel.json              # Vercel deployment config
├── package.json             # Project metadata
├── public/
│   ├── index.html           # Single-page research site
│   └── images/
│       ├── mie_efficiencies.png
│       ├── phase_functions.png
│       ├── angstrom_exponent.png
│       ├── size_distributions.png
│       ├── hygroscopic_growth.png
│       ├── sensor_heatmap.png
│       ├── ml_results.png
│       ├── model_dashboard.png
│       ├── feature_importance.png
│       ├── ablation_study.png
│       └── humidity_analysis.png
```

## Features
- Dark "Optical Observatory" theme matching the laser/photonics subject matter
- All 11 simulation result plots embedded with detailed captions
- Lightbox zoom on all figures
- Scroll-triggered reveal animations
- Responsive mobile layout
- Sticky navigation with active section highlighting
- No build step required — pure HTML/CSS/JS

## Team
- **Dishant Gupta** — System Design, Simulation & ML
- **Akanksha Verma** — Optical Design & Validation
- **Devashish** — Electronics & Firmware
- **Ajay Mokta** — Testing & Deployment
- **Dr. Arvind K. Gathania** — Project Supervisor
