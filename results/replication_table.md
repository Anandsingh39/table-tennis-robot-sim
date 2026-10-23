# Replication table (Phase 7)

Our numbers: Monte Carlo baseline (sigma 2 mm). Paper numbers are quoted from the papers (config/literature.yaml).
Differences to keep in mind: the papers are real robots with vision (latency, calibration error, ball spin,
unknown restitution); ours is a simulation with a specified Gaussian noise model and a known (or deliberately
mismatched) ball model.

| | This project (sim) | Nakashima et al. 2008 [3] | Ji et al. 2024 [7] |
|---|---|---|---|
| Platform | sim: OpenArm v2 (7 DOF, torque control) in MuJoCo | DENSO 6-joint manipulator, joint J4 fixed (5 DOF used); torque-input via D/A (Sec. 2) | 3-DOF parallel mechanism with ball-joint end effector; servo motors 4 N·m rated / 12 N·m peak, gear ratio 7.75 (Sec. 6) |
| Paddle | disk r = 75 mm, 0.15 kg | aluminium square plate 150 mm, 3 mm thick, 0.20 kg (Sec. 2) | hexagonal aluminium juggling paddle (Sec. 6) |
| Ball sensing | sim: position + N(0, (2 mm)²) at 200 Hz, EKF | 2 CCD cameras + QuickMAG tracker, 120 Hz sampling (Sec. 2) | 2 industrial cameras at 250 FPS, mounted 1 m above the robot (Secs. 1, 6) |
| Ball / impact model | gravity + quadratic drag; impulse, e = 0.8 (assumed) | no air resistance; isotropic restitution matrix at impact (Sec. 3.3, Assumptions 1-2); restitution values not stated | V_z,out = k V_z,in + b with k = -0.95, b = 0 (Eq. 13) |
| Control law | impact-targeting (mirror-law style) + IK + quintic + computed torque | racket xy tracks the ball, z mirrors the ball about the hitting plane; racket tilt regulates the hitting point; energy-based gain for ball height (Sec. 5) | PD controller on the paddle tilt about X and Y driven by the ball's distance from the paddle centre, output limited; T-shaped velocity profile with stroke delta_z = 0.037 z_h (Sec. 5) |
| Target | apex 40 cm above the hit plane; centre of the hit plane | hitting point [0.35, -0.65, 0.35] m (Sec. 6); apex target expressed through an energy threshold, 0.070 m (Sec. 6) | bounce heights 10, 30, 50 cm above the paddle; target point = paddle centre (Sec. 6) |
| Duration / length | 100% of 50 runs reached the 200-bounce cap (~114 s each) | one continuous 100 s run (Sec. 6, Fig. 12); number of hits not reported | experiment 1: 100 juggles per height; experiment 2: 30 s per height (Sec. 6) |
| Hitting-position error | impact RMS 7.1 mm, p95 12.3 mm (true state: 2.5 mm) | mean hitting-position error 1.4 cm (x) and 3.6 cm (y); most of the ball path inside a 5 cm-radius circle (Sec. 6, Fig. 12) | maximum landing errors 250, 2500, 6000 mm^2 at 10, 30, 50 cm (Sec. 6, Fig. 7; units as reported); dispersion grows with height |
| Height error | apex error -0.85 ± 1.05 mm, |err| p95 2.60 mm = 0.65% of the 40 cm target | not reported numerically | each juggle within 10% of the set height (Sec. 6, Figs. 9-10) |

Sources:
- [3] A. Nakashima, Y. Sugiyama, Y. Hayakawa, 'Paddle Juggling by Robot Manipulator with Visual Servo', in Robot Manipulators, InTech 2008, doi:10.5772/6218 (journal version of ICARCV 2006)
- [7] Y. Ji, B. Zhang, Y. Mao, H. Wang, X. Hu, L. Zhang, 'Design, Modeling, and Experimental Validation of a Vision-Based Table Tennis Juggling Robot', Mathematics 12(11):1634, 2024, doi:10.3390/math12111634
