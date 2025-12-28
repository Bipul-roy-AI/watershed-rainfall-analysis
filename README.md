# 🌧️ Advanced Watershed Rainfall Analysis Tool

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An automated geospatial tool designed to extract, analyze, and visualize monthly rainfall data for specific watershed regions. [cite_start]Built with **Streamlit** and **Python**, this application automates the process of Zonal Statistics, ensuring accurate hydrological analysis through rigorous data validation and CRS reprojection.

## 🚀 Key Features

* [cite_start]**Interactive Watershed Selection:** Upload shapefiles and dynamically select specific Upazilas or regions for analysis[cite: 98, 99].
* [cite_start]**Automated Zonal Statistics:** Computes Total, Mean, Min, and Max rainfall for selected geometries[cite: 126, 127].
* [cite_start]**Robust Data Validation:** Automatically checks for multi-band errors, negative values, and coordinate system mismatches before processing[cite: 78, 106].
* [cite_start]**DEM Integration:** (Optional) Upload Digital Elevation Models to calculate watershed elevation statistics (Min/Max/Mean/Relief)[cite: 90].
* [cite_start]**Dynamic Visualization:** Generates instant bar charts, trend lines, and downloadable statistical reports (CSV)[cite: 148, 150].

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/watershed-rainfall-analyzer.git](https://github.com/yourusername/watershed-rainfall-analyzer.git)
    cd watershed-rainfall-analyzer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Follow the workflow:**
    * [cite_start]**Step 1:** Upload your Watershed Shapefile (must be a `.zip` containing .shp, .shx, .dbf, .prj)[cite: 93].
    * [cite_start]**Step 2:** Select your region of interest from the dropdown menu[cite: 99].
    * **Step 3:** Upload monthly rainfall raster files (`.tif`). [cite_start]Ensure files are named with dates (e.g., `01-2022.tif`) for chronological sorting[cite: 105].
    * [cite_start]**Step 4:** Click **Run Rainfall Analysis**[cite: 115].

## 📂 Data Requirements

* [cite_start]**Shapefile:** Must be a zipped archive containing at least the `.shp`, `.shx`, and `.dbf` components[cite: 93].
* **Rainfall Data:** Single-band GeoTIFFs (`.tif`) representing rainfall depth.
* [cite_start]**Projections:** The app automatically handles CRS reprojection, but input data should ideally be georeferenced[cite: 124, 170].

## 📊 Outputs

The tool provides:
* [cite_start]**Summary Metrics:** Total and average rainfall, wettest/driest months[cite: 141].
* [cite_start]**Visualizations:** Bar charts and trend lines[cite: 152].
* [cite_start]**Raw Data:** A downloadable CSV file containing pixel counts and statistics for every month processed[cite: 150].

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Author:** Bipul Roy  
*GIS Analyst & Researcher*
