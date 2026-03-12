#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║          MINI-LIDAR AIR QUALITY SENSOR — SIMULATION FRAMEWORK      ║
║                                                                      ║
║  Multi-wavelength (450/532/650 nm) × Multi-angle (30°/60°/90°)      ║
║  Mie scattering simulation with ML-based PM2.5/PM10 calibration     ║
║                                                                      ║
║  B.Tech Major Project · NIT Hamirpur · Physics Department            ║
║  Supervisor: Dr. Arvind K. Gathania                                  ║
║  Team: Dishant Gupta, Akanksha Verma, Devashish, Ajay Mokta          ║
╚══════════════════════════════════════════════════════════════════════╝

Run:  python run_simulation.py [--samples N] [--skip-plots] [--quick]

This script executes the complete simulation pipeline:
    1. Mie scattering physics validation
    2. Aerosol size distribution generation
    3. Hygroscopic growth modelling
    4. Synthetic sensor data generation (9-channel + noise)
    5. Feature engineering (34 features)
    6. ML model training (RF, XGBoost, NN)
    7. Ablation study & humidity analysis
    8. Visualization (all plots saved to results/)

CPU-only. Python ≥ 3.10. No GPU required.
"""

import argparse
import os
import sys
import time
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    WAVELENGTHS_NM, WAVELENGTHS_UM, DETECTOR_ANGLES_DEG,
    REFRACTIVE_INDICES, N_SAMPLES, RANDOM_STATE
)


def banner(text: str):
    """Print a section banner."""
    w = 70
    print(f"\n{'═' * w}")
    print(f"  {text}")
    print(f"{'═' * w}")


def step(text: str):
    """Print a step label."""
    print(f"\n  ▶ {text}")


def main():
    parser = argparse.ArgumentParser(description="Mini-LIDAR Simulation Framework")
    parser.add_argument("--samples", type=int, default=N_SAMPLES,
                        help=f"Number of synthetic samples (default: {N_SAMPLES})")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with 500 samples, reduced resolution")
    args = parser.parse_args()
    
    if args.quick:
        args.samples = 500
    
    np.random.seed(RANDOM_STATE)
    t_start = time.time()
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 1: MIE SCATTERING PHYSICS VALIDATION                    ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    
    banner("PHASE 1: MIE SCATTERING PHYSICS")
    
    from core.mie_scattering import MieScatteringEngine
    engine = MieScatteringEngine()
    
    m_as = REFRACTIVE_INDICES["ammonium_sulfate"]
    
    # 1a. Single particle scattering at detector angles
    step("Computing single-particle scattering (d=1.0 µm, ammonium sulfate)")
    signals = engine.compute_scattering(1.0, m=m_as)
    print(f"    Scattering matrix (µm²/sr):")
    print(f"    {'':>12s}  {'30°':>12s}  {'60°':>12s}  {'90°':>12s}")
    for i, wl in enumerate(WAVELENGTHS_NM):
        print(f"    {wl} nm:   {signals[i,0]:12.4e}  {signals[i,1]:12.4e}  {signals[i,2]:12.4e}")
    
    # 1b. Mie efficiencies vs diameter
    step("Computing Mie efficiencies across particle sizes")
    d_range = np.logspace(-1.5, 1.2, 150 if not args.quick else 60)
    
    efficiencies = {}
    for wl_nm, wl_um in zip(WAVELENGTHS_NM, WAVELENGTHS_UM):
        qsca_arr = []
        qext_arr = []
        for d in d_range:
            eff = engine.compute_efficiencies(d, wl_um, m_as)
            qsca_arr.append(eff["Qsca"])
            qext_arr.append(eff["Qext"])
        efficiencies[wl_nm] = {"Qsca": np.array(qsca_arr), "Qext": np.array(qext_arr)}
    print(f"    Computed for {len(d_range)} diameters × 3 wavelengths")
    
    # 1c. Ångström exponent
    step("Computing Ångström exponent vs diameter")
    angstrom_diameters = np.logspace(np.log10(0.05), np.log10(15), 100 if not args.quick else 40)
    angstrom_values = np.array([
        engine.compute_angstrom_exponent(d, m_as) for d in angstrom_diameters
    ])
    print(f"    α(0.1 µm) = {angstrom_values[5]:.2f} (fine)")
    print(f"    α(1.0 µm) = {angstrom_values[50 if not args.quick else 20]:.2f} (transition)")
    print(f"    α(5.0 µm) = {angstrom_values[-10]:.2f} (coarse)")
    
    # 1d. Phase functions
    step("Computing phase functions for key particle sizes")
    phase_data = {}
    for d_um in [0.2, 0.5, 1.0, 2.5, 5.0]:
        theta, SL, SR, SU = engine.compute_phase_function(d_um, 0.532, m_as)
        phase_data[f"d = {d_um} µm"] = {"theta": theta, "SU": SU, "SL": SL, "SR": SR}
    print(f"    Phase functions computed for 5 particle sizes at 532 nm")
    
    # 1e. Cross-validation
    step("Cross-validating PyMieScatt vs miepython")
    for d in [0.5, 1.0, 2.0]:
        cv = engine.cross_validate_miepython(d, 0.532, m_as)
        print(f"    d={d} µm: Qext error = {cv['relative_error']['Qext']:.2e}, "
              f"Qsca error = {cv['relative_error']['Qsca']:.2e}")
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 2: AEROSOL DISTRIBUTION GENERATION                       ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    
    banner("PHASE 2: AEROSOL SIZE DISTRIBUTIONS")
    
    from core.aerosol_generator import AerosolGenerator
    aerosol_gen = AerosolGenerator()
    
    step("Generating distributions for each scenario")
    distributions = {}
    for scenario in ["urban", "biomass", "dust", "clean", "mixed"]:
        diameters, dNdlogD, pm25, pm10 = aerosol_gen.generate_sample(
            pm25_target=50.0, scenario=scenario
        )
        distributions[scenario] = dNdlogD
        ratio = pm25 / pm10 if pm10 > 0 else 0
        print(f"    {scenario:>10s}:  PM2.5={pm25:7.1f}  PM10={pm10:7.1f}  ratio={ratio:.2f}")
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 3: HYGROSCOPIC GROWTH MODELLING                          ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    
    banner("PHASE 3: HYGROSCOPIC GROWTH")
    
    from core.hygroscopic_growth import HygroscopicGrowthModel
    hgm = HygroscopicGrowthModel()
    
    step("Computing growth factors and scattering enhancement")
    rh_range = np.arange(10, 96, 2)
    growth_data = {}
    
    for atype in ["ammonium_sulfate", "organic_carbon", "sea_salt", "black_carbon"]:
        g_arr = [hgm.growth_factor(rh, aerosol_type=atype) for rh in rh_range]
        f_rh_arr = [hgm.scattering_enhancement(0.5, rh, aerosol_type=atype) for rh in rh_range]
        growth_data[atype] = {
            "growth_factor": np.array(g_arr),
            "scattering_enhancement": np.array(f_rh_arr),
        }
        print(f"    {atype:>20s}: g(80%)={g_arr[35]:.3f}, f(80%)={f_rh_arr[35]:.1f}×")
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 4: SYNTHETIC DATASET GENERATION                          ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    
    banner(f"PHASE 4: SYNTHETIC DATASET ({args.samples} samples)")
    
    from core.sensor_model import SensorModel
    from core.feature_engineering import FeatureEngineer
    
    sensor = SensorModel()
    feat_eng = FeatureEngineer()
    
    step("Generating physics-based synthetic sensor readings")
    t_gen = time.time()
    
    all_features = []
    all_pm25 = []
    all_pm10 = []
    all_rh = []
    
    scenarios = ["urban", "biomass", "dust", "clean", "mixed"]
    progress_interval = max(args.samples // 10, 1)
    
    for idx in range(args.samples):
        # Random conditions
        scenario = np.random.choice(scenarios, p=[0.35, 0.20, 0.15, 0.15, 0.15])
        pm25_target = np.random.uniform(5, 500)
        temperature = np.random.uniform(5, 45)
        rh = np.random.uniform(15, 95)
        
        # Generate aerosol distribution
        diameters, dNdlogD, pm25, pm10 = aerosol_gen.generate_sample(
            pm25_target=pm25_target, scenario=scenario
        )
        
        # Apply hygroscopic growth
        d_wet_scalar, m_wet = hgm.grow_particle(0.5, rh, aerosol_type="ammonium_sulfate")
        g_factor = hgm.growth_factor(rh, aerosol_type="ammonium_sulfate")
        
        # Compute scattering at representative diameters (fast approximation)
        # Instead of integrating over 200 bins, use weighted-average approach
        # Weight by dN/dlogD to find effective scattering diameter
        weights = dNdlogD + 1e-30
        d_eff = np.exp(np.average(np.log(diameters + 1e-30), weights=weights))
        d_eff_wet = d_eff * g_factor
        
        # Use 20 representative bins spanning the distribution's active range
        active_mask = dNdlogD > (dNdlogD.max() * 0.001)
        if active_mask.sum() > 20:
            active_d = diameters[active_mask]
            active_dN = dNdlogD[active_mask]
            # Subsample
            step_size = max(1, len(active_d) // 20)
            sub_d = active_d[::step_size]
            sub_dN = active_dN[::step_size]
        else:
            sub_d = diameters[active_mask] if active_mask.any() else diameters[:5]
            sub_dN = dNdlogD[active_mask] if active_mask.any() else dNdlogD[:5]
        
        sub_d_wet = sub_d * g_factor
        
        scattering = engine.compute_scattering_distribution(
            sub_d_wet, sub_dN, m=m_wet
        )
        
        # Scale scattering by PM concentration for a physically meaningful signal
        # Higher PM → more particles → more scattering
        pm_scale = pm25 / 50.0  # normalise around typical 50 µg/m³
        scattering *= pm_scale
        
        # Number density proportional to PM concentration
        n_density = pm25 * 2e7  # rough scaling: 50 µg/m³ → 1e9 #/m³
        
        # Simulate sensor electronics (with noise)
        voltages = sensor.simulate_reading_fast(scattering, n_density)
        
        # Extract features
        features, feat_names = feat_eng.extract(voltages, temperature, rh)
        
        all_features.append(features)
        all_pm25.append(pm25)
        all_pm10.append(pm10)
        all_rh.append(rh)
        
        if (idx + 1) % progress_interval == 0:
            pct = (idx + 1) / args.samples * 100
            elapsed = time.time() - t_gen
            rate = (idx + 1) / elapsed
            eta = (args.samples - idx - 1) / rate
            print(f"    [{pct:5.1f}%] {idx+1}/{args.samples} samples  "
                  f"({rate:.0f} samples/s, ETA: {eta:.0f}s)")
    
    X = np.array(all_features)
    y_pm25 = np.array(all_pm25)
    y_pm10 = np.array(all_pm10)
    rh_arr = np.array(all_rh)
    
    gen_time = time.time() - t_gen
    print(f"\n    Dataset generated: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"    Generation time: {gen_time:.1f}s ({X.shape[0]/gen_time:.0f} samples/s)")
    
    # Save dataset
    np.savez_compressed(
        "data/synthetic_dataset.npz",
        X=X, y_pm25=y_pm25, y_pm10=y_pm10, rh=rh_arr,
        feature_names=feat_names
    )
    print(f"    Saved to data/synthetic_dataset.npz")
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 5: ML MODEL TRAINING                                     ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    
    banner("PHASE 5: ML MODEL TRAINING")
    
    from ml.train_models import MiniLidarMLPipeline
    
    pipeline = MiniLidarMLPipeline(output_dir="results")
    
    step("Loading data and splitting train/test")
    pipeline.load_data(X, y_pm25, y_pm10, rh_values=rh_arr, feature_names=feat_names)
    
    step("Training all models")
    results = pipeline.train_all()
    
    step("Cross-validation (5-fold)")
    cv_results = pipeline.cross_validate()
    
    step("Feature importance analysis")
    importances = pipeline.feature_importance("pm25")
    
    step("Ablation study")
    ablation = pipeline.ablation_study("pm25")
    
    step("Humidity compensation analysis")
    humidity = pipeline.humidity_analysis("pm25")
    
    step("Saving results")
    pipeline.save_results()
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 6: VISUALIZATION                                         ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    
    if not args.skip_plots:
        banner("PHASE 6: GENERATING PLOTS")
        
        from visualization import (
            plot_size_distribution, plot_mie_efficiencies,
            plot_phase_functions, plot_hygroscopic_growth,
            plot_ml_results, plot_feature_importance,
            plot_angstrom_analysis, plot_sensor_heatmap,
            plot_ablation_study, plot_humidity_analysis,
            plot_model_comparison_dashboard
        )
        
        step("Plotting size distributions")
        plot_size_distribution(diameters, distributions)
        
        step("Plotting Mie efficiencies")
        plot_mie_efficiencies(d_range, efficiencies)
        
        step("Plotting phase functions")
        plot_phase_functions(None, phase_data)
        
        step("Plotting hygroscopic growth")
        plot_hygroscopic_growth(rh_range, growth_data)
        
        step("Plotting Ångström exponent")
        plot_angstrom_analysis(angstrom_diameters, angstrom_values)
        
        step("Plotting feature importance")
        plot_feature_importance(importances)
        
        step("Plotting ML results")
        y_preds = {}
        for name in results.keys():
            model_key = f"{name}_pm25"
            if model_key in pipeline.models:
                y_preds[name] = pipeline.models[model_key].predict(pipeline.X_test)
        plot_ml_results(results, pipeline.y_test["pm25"], y_preds)
        
        step("Plotting sensor signal heatmap & correlations")
        plot_sensor_heatmap(X, feat_names)
        
        step("Plotting ablation study")
        plot_ablation_study(ablation)
        
        step("Plotting humidity compensation analysis")
        plot_humidity_analysis(humidity)
        
        step("Plotting model comparison dashboard")
        plot_model_comparison_dashboard(results)
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  SUMMARY                                                        ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    
    banner("SIMULATION COMPLETE")
    
    total_time = time.time() - t_start
    
    print(f"\n  Total runtime: {total_time:.1f}s")
    print(f"  Samples generated: {args.samples}")
    print(f"  Features per sample: {X.shape[1]}")
    print()
    print(f"  ┌{'─' * 55}┐")
    print(f"  │  {'Model':>20s}  {'PM2.5 MAE':>12s}  {'PM2.5 R²':>12s}  │")
    print(f"  ├{'─' * 55}┤")
    for name, metrics in results.items():
        pm25 = metrics["pm25"]
        print(f"  │  {name:>20s}  {pm25['mae']:>10.2f}  {pm25['r2']:>12.4f}  │")
    print(f"  └{'─' * 55}┘")
    print()
    print(f"  Output files:")
    print(f"    data/synthetic_dataset.npz")
    print(f"    results/training_results.json")
    if not args.skip_plots:
        for f in sorted(os.listdir("results")):
            if f.endswith(".png"):
                print(f"    results/{f}")
    
    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()
