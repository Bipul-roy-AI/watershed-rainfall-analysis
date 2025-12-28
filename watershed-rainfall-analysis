import streamlit as st
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import tempfile
import os
import zipfile
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Watershed Rainfall Analysis", page_icon="🌧️", layout="wide")

st.title("🌧️ Advanced Watershed Rainfall Analysis Tool")
st.markdown("""
This application analyzes rainfall data for watersheds using **Zonal Statistics** with advanced validation and DEM support.

### 📋 Quick Start Guide:
1. Upload your **Watershed Shapefile** (as .zip containing .shp, .shx, .dbf, .prj)
2. **(Optional)** Upload **DEM** for watershed delineation
3. Select the **Region/Upazila** you want to analyze
4. Upload **Monthly Rainfall Rasters** (.tif files)
5. Click **Run Analysis** and view results!
""")

# --- Helper Functions ---
@st.cache_data
def load_shapefile(uploaded_zip):
    """Extract and load shapefile from ZIP"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "shapefile.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        
        for root, dirs, files in os.walk(tmpdirname):
            for file in files:
                if file.endswith(".shp"):
                    return gpd.read_file(os.path.join(root, file))
    return None

def validate_raster(tif_path, file_name):
    """
    Validate raster file for rainfall analysis
    Returns: (is_valid, message, metadata_dict)
    """
    try:
        with rasterio.open(tif_path) as src:
            # Check band count
            if src.count != 1:
                return False, f"❌ {file_name}: Multi-band raster detected ({src.count} bands). Rainfall rasters should have only 1 band.", None
            
            # Get metadata
            metadata = {
                'bands': src.count,
                'width': src.width,
                'height': src.height,
                'crs': src.crs,
                'nodata': src.nodata,
                'dtype': src.dtypes[0],
                'resolution': src.res
            }
            
            # Check for valid CRS
            if src.crs is None:
                return False, f"⚠️ {file_name}: No CRS defined. This may cause issues.", metadata
            
            # Check data type
            if src.dtypes[0] not in ['float32', 'float64', 'int16', 'int32', 'uint16', 'uint32']:
                return False, f"⚠️ {file_name}: Unusual data type ({src.dtypes[0]}). Expected numeric type.", metadata
            
            # Read data and check for valid values
            data = src.read(1)
            if src.nodata is not None:
                valid_data = data[data != src.nodata]
            else:
                valid_data = data[~np.isnan(data)]
            
            if len(valid_data) == 0:
                return False, f"❌ {file_name}: No valid data found (all NoData or NaN).", metadata
            
            # Check for negative values (rainfall shouldn't be negative)
            if np.any(valid_data < 0):
                return False, f"⚠️ {file_name}: Contains negative values. Rainfall should be positive.", metadata
            
            return True, f"✅ {file_name}: Valid", metadata
            
    except Exception as e:
        return False, f"❌ {file_name}: Error reading file - {str(e)}", None

def check_raster_consistency(metadata_list):
    """
    Check if all rasters have consistent resolution and CRS
    Returns: (is_consistent, message)
    """
    if not metadata_list:
        return False, "No valid rasters to compare"
    
    first = metadata_list[0]
    issues = []
    
    for i, meta in enumerate(metadata_list[1:], 1):
        # Check resolution
        if meta['resolution'] != first['resolution']:
            issues.append(f"Raster {i+1} has different resolution: {meta['resolution']} vs {first['resolution']}")
        
        # Check CRS
        if meta['crs'] != first['crs']:
            issues.append(f"Raster {i+1} has different CRS: {meta['crs']} vs {first['crs']}")
    
    if issues:
        return False, "\n".join(issues)
    
    return True, f"✅ All rasters have consistent resolution ({first['resolution']}) and CRS ({first['crs']})"

def process_dem(dem_path):
    """
    Process DEM to extract basic watershed information
    Returns: (success, message, stats_dict)
    """
    try:
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            
            # Basic statistics
            valid_data = dem_data[dem_data != src.nodata] if src.nodata else dem_data[~np.isnan(dem_data)]
            
            stats = {
                'min_elevation': float(np.min(valid_data)),
                'max_elevation': float(np.max(valid_data)),
                'mean_elevation': float(np.mean(valid_data)),
                'elevation_range': float(np.max(valid_data) - np.min(valid_data)),
                'crs': src.crs,
                'resolution': src.res
            }
            
            return True, "✅ DEM processed successfully", stats
            
    except Exception as e:
        return False, f"❌ Error processing DEM: {str(e)}", None

# --- Step 1: Upload Shapefile ---
st.header("1️⃣ Upload Watershed Boundary")
uploaded_zip = st.file_uploader(
    "Upload Shapefile (as .zip)", 
    type="zip",
    help="ZIP file must contain .shp, .shx, .dbf, and .prj files"
)

# --- Step 1.5: Optional DEM Upload ---
st.header("1.5️⃣ (Optional) Upload DEM for Watershed Analysis")
uploaded_dem = st.file_uploader(
    "Upload Digital Elevation Model (.tif)", 
    type=["tif", "tiff"],
    help="Optional: Upload DEM for elevation analysis and watershed characteristics"
)

dem_stats = None
if uploaded_dem:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dem_path = os.path.join(tmp_dir, "dem.tif")
        with open(dem_path, "wb") as f:
            f.write(uploaded_dem.getbuffer())
        
        success, message, dem_stats = process_dem(dem_path)
        
        if success:
            st.success(message)
            
            # Display DEM statistics
            with st.expander("📊 View DEM Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min Elevation", f"{dem_stats['min_elevation']:.2f} m")
                with col2:
                    st.metric("Max Elevation", f"{dem_stats['max_elevation']:.2f} m")
                with col3:
                    st.metric("Mean Elevation", f"{dem_stats['mean_elevation']:.2f} m")
                with col4:
                    st.metric("Elevation Range", f"{dem_stats['elevation_range']:.2f} m")
                
                st.info(f"**DEM CRS:** {dem_stats['crs']}")
                st.info(f"**Resolution:** {dem_stats['resolution'][0]:.2f} x {dem_stats['resolution'][1]:.2f}")
        else:
            st.error(message)

if uploaded_zip is not None:
    try:
        with st.spinner("Loading shapefile..."):
            gdf = load_shapefile(uploaded_zip)
        
        st.success(f"✅ Shapefile loaded successfully! ({len(gdf)} regions found)")
        st.info(f"**Coordinate System:** {gdf.crs}")
        
        # --- Display Shapefile Preview ---
        with st.expander("📊 View All Regions Data"):
            st.dataframe(gdf.drop(columns='geometry'), use_container_width=True)
        
        # --- Step 2: Select Region ---
        st.header("2️⃣ Select Region of Interest")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Let user choose identifier column
            col_name = st.selectbox(
                "Choose column to identify regions:", 
                gdf.columns.drop('geometry'),
                help="Select the column that contains region names (e.g., 'Upazila', 'Name', 'ID')"
            )
            
            # Let user select specific region
            unique_values = sorted(gdf[col_name].unique())
            selected_region_name = st.selectbox(
                f"Select {col_name}:", 
                unique_values
            )
            
            # Filter selected polygon
            selected_poly = gdf[gdf[col_name] == selected_region_name].copy()
            
            # Display region info
            st.markdown(f"**Selected:** {selected_region_name}")
            if 'Area' in selected_poly.columns:
                area = selected_poly['Area'].values[0]
                st.markdown(f"**Area:** {area:,.2f} sq. meters")
        
        with col2:
            # Show map preview
            st.markdown("**Selected Region Map:**")
            fig_map, ax_map = plt.subplots(figsize=(8, 6))
            gdf.plot(ax=ax_map, color='lightgray', edgecolor='black', alpha=0.5)
            selected_poly.plot(ax=ax_map, color='steelblue', edgecolor='darkblue', linewidth=2)
            ax_map.set_title(f"Selected: {selected_region_name}")
            ax_map.axis('off')
            st.pyplot(fig_map)
        
        # --- Step 3: Upload Rasters ---
        st.header("3️⃣ Upload Rainfall Data")
        uploaded_tifs = st.file_uploader(
            "Upload Monthly Rainfall Rasters (.tif)", 
            type=["tif", "tiff"], 
            accept_multiple_files=True,
            help="Upload one or more GeoTIFF files. Name them with dates (e.g., '01-2022.tif')"
        )
        
        if uploaded_tifs:
            st.success(f"✅ {len(uploaded_tifs)} raster file(s) uploaded")
            
            # --- Validation Section ---
            with st.expander("🔍 Click to Validate Rasters Before Analysis"):
                if st.button("Run Validation Checks"):
                    validation_results = []
                    valid_metadata = []
                    
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        for tif_file in uploaded_tifs:
                            tif_path = os.path.join(tmp_dir, tif_file.name)
                            with open(tif_path, "wb") as f:
                                f.write(tif_file.getbuffer())
                            
                            is_valid, msg, metadata = validate_raster(tif_path, tif_file.name)
                            validation_results.append((is_valid, msg))
                            
                            if is_valid and metadata:
                                valid_metadata.append(metadata)
                    
                    # Display validation results
                    st.markdown("### Validation Results:")
                    for is_valid, msg in validation_results:
                        if "✅" in msg:
                            st.success(msg)
                        elif "⚠️" in msg:
                            st.warning(msg)
                        else:
                            st.error(msg)
                    
                    # Check consistency
                    if valid_metadata:
                        is_consistent, consistency_msg = check_raster_consistency(valid_metadata)
                        if is_consistent:
                            st.success(consistency_msg)
                        else:
                            st.warning(f"⚠️ Consistency Issues:\n{consistency_msg}")
                    
                    # Summary
                    valid_count = sum(1 for v, _ in validation_results if v)
                    st.info(f"**Summary:** {valid_count}/{len(validation_results)} files passed validation")
            
            # --- Analysis Button ---
            if st.button("🚀 Run Rainfall Analysis", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Track validation issues during processing
                skipped_files = []
                
                # Process rasters
                with tempfile.TemporaryDirectory() as raster_tmp:
                    for i, tif_file in enumerate(uploaded_tifs):
                        status_text.text(f"Processing: {tif_file.name} ({i+1}/{len(uploaded_tifs)})")
                        progress_bar.progress((i + 1) / len(uploaded_tifs))
                        
                        # Save TIF to temp path
                        tif_path = os.path.join(raster_tmp, tif_file.name)
                        with open(tif_path, "wb") as f:
                            f.write(tif_file.getbuffer())
                        
                        try:
                            # Quick validation
                            is_valid, msg, metadata = validate_raster(tif_path, tif_file.name)
                            if not is_valid:
                                skipped_files.append((tif_file.name, msg))
                                continue
                            
                            # Get raster CRS
                            with rasterio.open(tif_path) as src:
                                raster_crs = src.crs
                                nodata_value = src.nodata
                            
                            # Reproject vector to match raster CRS (CRITICAL for accuracy!)
                            if gdf.crs != raster_crs:
                                process_poly = selected_poly.to_crs(raster_crs)
                            else:
                                process_poly = selected_poly
                            
                            # Calculate Zonal Statistics with proper NoData handling
                            stats = zonal_stats(
                                process_poly,
                                tif_path,
                                stats=['sum', 'mean', 'min', 'max', 'count'],
                                nodata=nodata_value,
                                geojson=True
                            )
                            
                            # Extract results
                            props = stats[0]['properties']
                            rainfall_sum = props.get('sum', 0)
                            rainfall_mean = props.get('mean', 0)
                            rainfall_min = props.get('min', 0)
                            rainfall_max = props.get('max', 0)
                            pixel_count = props.get('count', 0)
                            
                            # Extract month from filename
                            month_name = os.path.splitext(tif_file.name)[0]
                            
                            results.append({
                                "Month": month_name,
                                "Total_Rainfall_mm": rainfall_sum if rainfall_sum else 0,
                                "Mean_Rainfall_mm": rainfall_mean if rainfall_mean else 0,
                                "Min_Rainfall_mm": rainfall_min if rainfall_min else 0,
                                "Max_Rainfall_mm": rainfall_max if rainfall_max else 0,
                                "Pixel_Count": pixel_count
                            })
                            
                        except Exception as e:
                            skipped_files.append((tif_file.name, f"Processing error: {str(e)}"))
                
                progress_bar.empty()
                status_text.empty()
                
                # Show skipped files if any
                if skipped_files:
                    with st.expander(f"⚠️ {len(skipped_files)} file(s) were skipped"):
                        for filename, reason in skipped_files:
                            st.warning(f"**{filename}**: {reason}")
                
                # --- Step 4: Display Results ---
                if results:
                    st.success("✅ Analysis Complete!")
                    
                    # Create DataFrame
                    df = pd.DataFrame(results)
                    
                    # Try to sort chronologically
                    try:
                        df['Date_Sort'] = pd.to_datetime(df['Month'], format='%m-%Y', errors='coerce')
                        if df['Date_Sort'].notna().any():
                            df = df.sort_values('Date_Sort').drop(columns=['Date_Sort'])
                    except:
                        pass
                    
                    # Display results
                    st.header("📊 Analysis Results")
                    
                    # Summary Statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Total Rainfall", f"{df['Total_Rainfall_mm'].sum():,.0f} mm")
                    with col_stat2:
                        st.metric("Average Monthly", f"{df['Total_Rainfall_mm'].mean():,.0f} mm")
                    with col_stat3:
                        max_month = df.loc[df['Total_Rainfall_mm'].idxmax(), 'Month']
                        max_val = df['Total_Rainfall_mm'].max()
                        st.metric("Wettest Month", max_month, f"{max_val:,.0f} mm")
                    with col_stat4:
                        min_month = df.loc[df['Total_Rainfall_mm'].idxmin(), 'Month']
                        min_val = df['Total_Rainfall_mm'].min()
                        st.metric("Driest Month", min_month, f"{min_val:,.0f} mm")
                    
                    # DEM Integration - Add elevation context if DEM was uploaded
                    if dem_stats:
                        st.markdown("---")
                        st.subheader("🏔️ Watershed Characteristics")
                        dem_col1, dem_col2, dem_col3 = st.columns(3)
                        with dem_col1:
                            st.info(f"**Elevation Range:** {dem_stats['min_elevation']:.0f} - {dem_stats['max_elevation']:.0f} m")
                        with dem_col2:
                            st.info(f"**Mean Elevation:** {dem_stats['mean_elevation']:.0f} m")
                        with dem_col3:
                            st.info(f"**Relief:** {dem_stats['elevation_range']:.0f} m")
                    
                    # Data Table and Visualizations
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📊 Bar Chart", "📈 Trend Line", "📉 Statistics"])
                    
                    with tab1:
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Download Results as CSV",
                            csv,
                            f"rainfall_analysis_{selected_region_name}.csv",
                            "text/csv"
                        )
                    
                    with tab2:
                        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
                        bars = ax_bar.bar(df['Month'], df['Total_Rainfall_mm'], 
                                         color='steelblue', edgecolor='navy', alpha=0.7)
                        
                        # Highlight max and min
                        max_idx = df['Total_Rainfall_mm'].idxmax()
                        min_idx = df['Total_Rainfall_mm'].idxmin()
                        bars[max_idx].set_color('darkgreen')
                        bars[min_idx].set_color('darkred')
                        
                        ax_bar.set_title(f"Monthly Rainfall - {selected_region_name}", 
                                        fontsize=14, fontweight='bold')
                        ax_bar.set_ylabel("Total Rainfall (mm)", fontsize=12)
                        ax_bar.set_xlabel("Month", fontsize=12)
                        plt.xticks(rotation=45, ha='right')
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_bar)
                    
                    with tab3:
                        fig_line, ax_line = plt.subplots(figsize=(12, 6))
                        ax_line.plot(df['Month'], df['Total_Rainfall_mm'], 
                                   marker='o', linewidth=2, markersize=8, 
                                   color='steelblue', markerfacecolor='orange')
                        ax_line.fill_between(range(len(df)), df['Total_Rainfall_mm'], 
                                           alpha=0.3, color='steelblue')
                        ax_line.set_title(f"Rainfall Trend - {selected_region_name}", 
                                        fontsize=14, fontweight='bold')
                        ax_line.set_ylabel("Total Rainfall (mm)", fontsize=12)
                        ax_line.set_xlabel("Month", fontsize=12)
                        plt.xticks(rotation=45, ha='right')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_line)
                    
                    with tab4:
                        st.markdown("### 📊 Detailed Statistics")
                        
                        # Statistical summary
                        stats_df = df[['Total_Rainfall_mm', 'Mean_Rainfall_mm', 'Min_Rainfall_mm', 'Max_Rainfall_mm']].describe()
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Additional insights
                        st.markdown("### 🔍 Insights")
                        total_variance = df['Total_Rainfall_mm'].var()
                        st.write(f"**Variance:** {total_variance:,.2f} mm²")
                        st.write(f"**Standard Deviation:** {df['Total_Rainfall_mm'].std():,.2f} mm")
                        st.write(f"**Coefficient of Variation:** {(df['Total_Rainfall_mm'].std() / df['Total_Rainfall_mm'].mean() * 100):.2f}%")
                
                else:
                    st.error("❌ No results generated. All files were skipped due to validation errors.")
        
        else:
            st.info("👆 Upload rainfall raster files to continue")
    
    except Exception as e:
        st.error(f"❌ Error loading shapefile: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload a shapefile ZIP to begin")
    
    st.markdown("""
    ---
    ### 💡 Tips for Best Results:
    
    **File Requirements:**
    - **Shapefile ZIP must contain:** `.shp`, `.shx`, `.dbf`, and `.prj` files
    - **Raster files:** Use GeoTIFF format (`.tif` or `.tiff`)
    - **Single-band rasters:** Rainfall data should be single-band only
    - **File naming:** Name files with dates (e.g., `01-2022.tif`, `02-2022.tif`)
    
    **Data Quality:**
    - **CRS matching:** The app automatically handles different coordinate systems
    - **Resolution consistency:** All rainfall rasters should have the same resolution
    - **NoData handling:** Properly set NoData values in your rasters
    - **Positive values:** Rainfall values should be non-negative
    
    **Optional DEM Analysis:**
    - Upload a DEM to get elevation statistics for your watershed
    - DEM should cover the same area as your watershed boundary
    - Useful for understanding topographic influences on rainfall
    
    ### 🛠️ Advanced Features:
    - ✅ **Automatic validation** of raster files
    - ✅ **CRS reprojection** for accurate analysis
    - ✅ **NoData handling** to exclude invalid pixels
    - ✅ **Multi-band detection** to prevent errors
    - ✅ **Resolution consistency checks**
    - ✅ **DEM integration** for watershed characteristics
    
    ### 📧 Need Help?
    Make sure your data is properly georeferenced and in a standard CRS (e.g., WGS84, UTM)
    """)

# Footer
st.markdown("---")
st.markdown("🌧️ **Advanced Watershed Rainfall Analysis Tool** | Made with Streamlit | v2.0")
