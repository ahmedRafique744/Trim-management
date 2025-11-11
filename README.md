# High-Accuracy Sensor Fusion for Maritime Navigation (ORBIS Internship)

This repository contains a summary and analysis of my work as a Data Science/Engineering Intern at ORBIS in Summer 2025.

The goal of this project was to solve a core R&D problem: processing noisy sensor data to enable a new, high-accuracy navigation product for the maritime industry.

---

## 1. The Business Problem: "Garbage In, Garbage Out"

The company was developing a "Trim Management System" to help ships estimate their position and orientation, with the goal of optimizing fuel usage and improving safety.

The product relied on data from IMU (Gyroscope, Accelerometer) and GPS sensors. [cite_start]However, the raw sensor data was extremely noisy, with "large data errors in all three axes"[cite: 60]. This made high-accuracy positioning impossible.

**This is what the raw, unusable data looked like:**

*<img width="1120" height="213" alt="image" src="https://github.com/user-attachments/assets/6363a687-b321-4768-ab12-291747cf1356" />
*

## 2. My Solution: A Multi-Stage Processing Pipeline

As an intern on the 5-person beta prototype team, I was responsible for the data analysis and sensor fusion development. I built a multi-stage pipeline in Python (NumPy) to transform this noisy data into a reliable, high-accuracy output.

My workflow involved four key steps:

1.  **Time Alignment & Signal Processing:** Implemented resampling and decimation (using `scipy.signal`) to align the different sensor data streams and reduce data spikes caused by measurement errors.
2.  **Frame of Reference Rotation:** Wrote functions to rotate data from the local IMU sensor frame to the Earth's (NED) frame. This involved implementing **quaternion mathematics** to apply the correct rotation matrices.
3.  **State Estimation (Kalman Filter):** Optimized a standard Kalman Filter to work with our specific data types (position/velocity vectors from GPS). This provided an initial accurate prediction of the ship's states.
4.  **High-Accuracy Filtering (Unscented Kalman Filter):** To achieve the highest accuracy possible, I was tasked with utilizing a more advanced **Unscented Kalman Filter (UKF)**. This more computationally advanced filter was able to "fairly accurate[ly]... predict... trailer positions using noisy GPS measurements".

## 3. The Results: Creating Confidence from Chaos

My work directly contributed to solving the core technical problem. The pipeline successfully filtered the noisy raw data and produced a smooth, accurate estimation of the ship's position and velocity.

The "before-and-after" plots below show the raw GPS data (blue) vs. the final, filtered output from my Kalman Filter (green/orange).

**This is the final, high-value result:**
*<img width="1099" height="207" alt="image" src="https://github.com/user-attachments/assets/767eb7eb-cc1b-41b0-9250-21232ea247aa" />
*

**Example 1: Position (X vs. Y)**
``

**Example 2: Velocity & Position (per axis)**
``

These graphs clearly show how the filter accurately estimates the true path, smoothing out the noise and providing the reliable data the product needed to function.

## 4. Tech Stack & Skills

* **Core:** Python
* **Scientific Computing:** NumPy, SciPy, Matplotlib
* **Algorithms:** Kalman Filters (KF, UKF), State Estimation , Optimization, Signal Processing (Resampling, Decimation), Quaternion Mathematics.
* **Data Analysis:** PCA, Time Series Analysis
* **Other:** C, 3D Modeling 
