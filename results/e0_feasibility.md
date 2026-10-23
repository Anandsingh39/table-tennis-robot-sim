# E0 rally feasibility

## 1. Required paddle state for cooperative rally shots

Hit at the juggling hit centre [0.475, 0.0, 1.0]; land mid far half; incoming = mirror shot after a bounce on our half.

| flight time [s] | v_out [m/s] | incoming ball at hit x: z [m], v_in [m/s] | paddle tilt toward opponent [deg] | paddle normal speed [m/s] |
|---|---|---|---|---|
| 0.6 | [4.36 0.   2.69] (|v| 5.13) | 1.15, [-2.79  0.   -0.2 ] | 68 | 1.63 |
| 0.8 | [3.35 0.   3.89] (|v| 5.13) | 1.33, [-2.   -0.   -0.68] | 50 | 1.95 |
| 1.0 | [2.79 0.   5.12] (|v| 5.83) | 1.47, [-1.49  0.   -1.41] | 33 | 2.34 |
| 1.2 | [2.46 0.   6.44] (|v| 6.90) | 1.48, [-1.11 -0.   -2.46] | 22 | 2.63 |

## 2. Reachable hit points for forward-tilted paddle poses

Grid x [0.15, 0.66], z [0.85, 1.46], step 0.05 m, y = 0; reachable = IK within 1 mm / 0.5 deg with every joint >= 0.15 rad from its limits.

| paddle tilt [deg] | mount pre-tilt 0 deg | mount pre-tilt 40 deg |
|---|---|---|
| 20 | 30 pts (z 0.85–1.10 m) | 38 pts (z 0.95–1.40 m) |
| 30 | 23 pts (z 0.85–1.10 m) | 43 pts (z 0.85–1.35 m) |
| 40 | 11 pts (z 0.85–0.95 m) | 34 pts (z 0.85–1.15 m) |
| 50 | 4 pts (z 0.85–0.95 m) | 27 pts (z 0.85–1.10 m) |
| 60 | 0 pts | 18 pts (z 0.85–1.05 m) |

## 3. Joint speeds and torques

- juggle_nominal: max |q̇| [0.35 0.43 0.13 2.37 0.36 2.17 0.17] rad/s = [ 3  4  3 63  3 17  1] % of rated
- juggle_noisy: max |q̇| [0.71 0.49 0.3  2.59 0.43 2.22 0.21] rad/s = [ 7  5  8 69  3 18  2] % of rated
- tracked rally stroke (30.0 deg tilt, 2.35 m/s at [0.5, 0.0, 0.9]): impact error 0.62 mm / 0.08 deg / -0.27 ms
  - max |q̇| [5.76 0.33 0.41 3.33 0.53 7.38 0.23] rad/s = [55  3 11 88  4 59  2] % of rated, [34  2  8 61  3 35  1] % of max
  - peak torque before impact [63  3  5 38  8 72  5] %, after impact (follow-through) [ 44   2   2  38   7 100   5] %; planned trajectory min joint margin -0.11 rad
