This Python script implements a localization algorithm designed for the **Huawei 2024 Wireless Algorithm Contest**. The goal of the contest is to enhance the accuracy of localization in a wireless environment using Channel Charting techniques. The algorithm processes channel data to predict the positions of users (UEs) based on their wireless signal characteristics, leveraging a small set of known positions (anchor points).

**Features:**
- **Input**: The script reads multiple files, including:
  - **Configuration Files**: Contain information about the base station, number of data points, antennas, subcarriers, etc.
  - **Anchor Position Files**: Contain the known positions of some users (anchors).
  - **Channel Data Files**: Contain the raw channel data collected from the base station.
  
- **Algorithm**: The core algorithm uses **matrix operations** to process the channel data (in the frequency and delay domains), calculate the predicted user positions, and adjust for any **Timing Advance (TA)** or noise present in the data.

- **Optimized for Large Datasets**: The script implements **parallel processing** to handle large channel datasets efficiently, as the total amount of data can reach tens of gigabytes.

- **Output**: The predicted 2D positions of the users are saved to output files, which can be compared to the actual ground truth for evaluation.

### Objective:
- The objective is to predict the 2D positions of users (e.g., mobile devices) from their channel data collected by the base station. The challenge involves overcoming issues like **low signal-to-noise ratio (SNR)**, **non-line-of-sight (NLOS)** scenarios, and **Timing Advance (TA)**, which distort the data. 

### How It Works:
1. **Data Preprocessing**: The script processes large channel data files by reading them in slices, efficiently extracting and reshaping the data for further analysis.
2. **Localization Calculation**: Using the **Channel Charting (CC)** approach, the algorithm maps the high-dimensional channel data to a lower-dimensional space. The **Euclidean distance** between users with similar channel characteristics is then used to estimate their relative positions.
3. **Parallelization**: The script takes advantage of **parallel processing** (via `ThreadPoolExecutor`) to process multiple datasets concurrently, significantly reducing the time needed to complete the localization task.
