#!/usr/bin/env python
# Earth Observation Random Forest Classification and Change Detection Script
# This script trains a Random Forest model on Sentinel-2 data for land cover classification
# and performs change detection between two dates using a hybrid approach:
# - StatAPI for training data from polygons
# - EOPatches for prediction on the entire area

# ===== IMPORTS =====
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sentinel Hub imports
from sentinelhub import (
    SentinelHubRequest, DataCollection, SHConfig,
    SentinelHubStatisticalDownloadClient, SentinelHubStatistical,
    CRS, BBox, Geometry, bbox_to_dimensions, MimeType, BBoxSplitter
)
from eolearn.core import EOPatch, FeatureType
from eolearn.io import SentinelHubInputTask

# Import utility functions
from util import stat_to_df

# ===== HELPER FUNCTIONS =====
def to_download_requests(gdf, aggregation, calculations, input_data, data_folder=None):
    """Create StatAPI request for each geometry in geopandas GeoDataFrame"""
    stat_requests = []
    for row in gdf.itertuples():
        req = SentinelHubStatistical(
            aggregation=aggregation, 
            calculations=calculations, 
            input_data=[input_data], 
            geometry=Geometry(row.geometry, crs=CRS(gdf.crs.to_epsg())),
            data_folder=data_folder
        )
        stat_requests.append(req)
    
    download_requests = [stat_request.download_list[0] for stat_request in stat_requests]
    
    if data_folder:
        for download_request in download_requests:
            download_request.save_response = True
    
    return download_requests

def prepare_features_from_stat_data(stat_data, gdf):
    """Extract features from statistical data"""
    features = []
    labels = []
    
    for idx, stat in enumerate(stat_data):
        # Convert to DataFrame
        df = stat_to_df(stat)
        
        if df.empty:
            continue
        
        # Extract features
        feature_dict = {}
        
        # Extract band values (using percentile 50 which is median)
        for band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']:
            col = f'bands_{band}_percentiles_50.0'
            if col in df.columns:
                feature_dict[band] = df[col].values[0] / 10000.0  # Convert to reflectance
        
        features.append(feature_dict)
        labels.append(gdf.iloc[idx]['type'])
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    
    # Handle missing values
    features_df = features_df.fillna(0)
    
    return features_df, labels

def get_bbox_from_gdf(gdf):
    """Get bounding box from GeoDataFrame"""
    # Convert to UTM coordinate system for better spatial analysis
    gdf_utm = gdf.to_crs(epsg=32633)  # UTM zone 33N
    
    # Get bounding box for the entire area
    minx, miny, maxx, maxy = gdf_utm.total_bounds
    bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS(gdf_utm.crs.to_epsg()))
    
    return bbox, gdf_utm

def extract_features_for_classification(eopatch, time_idx=0):
    """Extract features from an EOPatch for classification"""
    # Get the dimensions of the EOPatch
    # The dataMask might have more than 2 dimensions, so we need to handle that
    mask_shape = eopatch.mask['dataMask'][time_idx].shape
    
    # If the mask has more than 2 dimensions, take the first 2
    if len(mask_shape) > 2:
        height, width = mask_shape[:2]
    else:
        height, width = mask_shape
    
    # Create feature array
    features = np.zeros((height * width, 12))  # 12 bands
    
    # Extract band values
    bands = eopatch.data['BANDS'][time_idx]
    for i in range(12):  # 12 bands
        features[:, i] = bands[:, :, i].flatten()
    
    # Create DataFrame with proper column names
    columns = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    features_df = pd.DataFrame(features, columns=columns)
    
    # Handle missing values
    features_df = features_df.fillna(0)
    
    return features_df, (height, width)

def predict_and_reshape(model, features_df, shape):
    """Make predictions and reshape to original image dimensions"""
    predictions = model.predict(features_df)
    return predictions.reshape(shape)

# ===== MAIN SCRIPT =====
if __name__ == "__main__":
    print("Starting Earth Observation Random Forest Classification and Change Detection...")
    
    # Set up Sentinel Hub configuration
    config = SHConfig()
    # Fill in your credentials here
    config.sh_client_id = ""
    config.sh_client_secret = ""
    config.instance_id = ""
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    
    # Define collections
    collection_l2a = DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url)
    
    # Load GeoJSON file containing polygons
    print("Loading GeoJSON data...")
    gdf = gpd.read_file('data/statapi_lausitz.geojson')
    # Convert to UTM coordinate system for better spatial analysis
    gdf_utm = gdf.to_crs(epsg=32633)  # UTM zone 33N
    print(f"Loaded {len(gdf_utm)} polygons")
    
    # Get bounding box for the entire area
    bbox, _ = get_bbox_from_gdf(gdf)
    print(f"Bounding box: {bbox}")
    
    # Define time intervals for both dates
    # 2024 data
    date_2024 = "20.8.2024"
    date_obj_2024 = datetime.strptime(date_2024, "%d.%m.%Y")
    next_day_2024 = date_obj_2024 + timedelta(days=1)
    time_interval_2024 = (date_obj_2024.strftime("%Y-%m-%d"), next_day_2024.strftime("%Y-%m-%d"))
    
    # 2016 data
    date_2016 = "27.8.2016"
    date_obj_2016 = datetime.strptime(date_2016, "%d.%m.%Y")
    next_day_2016 = date_obj_2016 + timedelta(days=1)
    time_interval_2016 = (date_obj_2016.strftime("%Y-%m-%d"), next_day_2016.strftime("%Y-%m-%d"))
    
    print(f"Time interval for 2024: {time_interval_2024}")
    print(f"Time interval for 2016: {time_interval_2016}")
    
    # ===== PART 1: Get training data using StatAPI =====
    print("Getting training data using StatAPI...")
    
    # Create evalscript for StatAPI
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{
          bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "dataMask"],
          units: "DN"
        }],
        output: [
          {
            id: "bands",
            bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
            sampleType: "UINT16"
          },
          {
            id: "dataMask",
            bands: 1
          }]
      }
    }
    
    function evaluatePixel(samples) {
        return {
            bands: [samples.B01, samples.B02, samples.B03, samples.B04, samples.B05, samples.B06, 
                    samples.B07, samples.B08, samples.B8A, samples.B09, samples.B11, samples.B12],
            dataMask: [samples.dataMask]
        };
    }
    """
    
    # Create aggregation, input data, and calculations for 2024
    aggregation = SentinelHubStatistical.aggregation(
        evalscript=evalscript,
        time_interval=time_interval_2024,
        aggregation_interval='P1D'
    )
    
    input_data = SentinelHubRequest.input_data(collection_l2a, maxcc=0.8)
    
    calculations = {
        "default": {
            "statistics": {
                "default": {
                    "percentiles": {
                        "k": [50]  # median value
                    }
                }
            }
        }
    }
    
    # Get data for 2024 using StatAPI
    print("Getting data for 20.8.2024 using StatAPI...")
    download_requests = to_download_requests(gdf_utm, aggregation, calculations, input_data, data_folder="./cache/")
    
    client = SentinelHubStatisticalDownloadClient(config=config)
    stat_data = client.download(download_requests)
    
    print(f"Retrieved data for {len(stat_data)} polygons")
    
    # Process data and extract features
    print("Extracting features from statistical data...")
    features_df, labels = prepare_features_from_stat_data(stat_data, gdf_utm)
    
    # Encode class labels as integers
    print("Encoding class labels...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(features_df, labels_encoded)  # Train on encoded labels
    
    print("Model trained successfully!")
    
    # train/test split code
    """
    # To split the data for training and testing:
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    
    # Train model on training data
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model on test data
    y_pred = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    """
    
    # ===== PART 2: Get prediction data using EOPatches =====
    print("Getting prediction data using EOPatches...")
    
    # Split the area into smaller bounding boxes using BBoxSplitter
    # Following the approach in 1_data_sources_explorer.ipynb
    print("Splitting area into smaller bounding boxes...")
    
    # Get the geometry from the GeoDataFrame
    aoi_shape = gdf_utm.unary_union.envelope
    
    # Split area into a 5x5 grid (adjust as needed)
    bbox_splitter = BBoxSplitter([aoi_shape], CRS.UTM_33N, (5, 5))
    bbox_list = bbox_splitter.get_bbox_list()
    
    # Get information about the grid
    info_list = bbox_splitter.get_info_list()
    
    # Select a single patch from the grid (e.g., the center one)
    # Find the center patch
    grid_size = 5
    selected_idx = 6 # center # (grid_size // 2) * grid_size + (grid_size // 2)
    # Make sure the index is valid
    if selected_idx >= len(bbox_list):
        print(f"Selected index {selected_idx} is out of range. Setting to 0.")
        selected_idx = 0
    
    selected_bbox = bbox_list[selected_idx]
    print(f"Selected patch {selected_idx} from the grid")
    
    # Create input task for EOPatches
    input_task = SentinelHubInputTask(
        data_collection=collection_l2a,
        bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
        bands_feature=(FeatureType.DATA, 'BANDS'),
        additional_data=[(FeatureType.MASK, 'dataMask')],
        resolution=(10, 10),  # 10m resolution
        maxcc=0.8,
        config=config
    )
    
    # Get data for 2024
    print("Getting data for 20.8.2024 using EOPatch...")
    eopatch_2024 = input_task.execute(bbox=selected_bbox, time_interval=time_interval_2024)
    
    # Get data for 2016
    print("Getting data for 27.8.2016 using EOPatch...")
    eopatch_2016 = input_task.execute(bbox=selected_bbox, time_interval=time_interval_2016)
    
    # Extract features for classification from both dates
    print("Extracting features for classification...")
    features_df_2024, shape_2024 = extract_features_for_classification(eopatch_2024)
    features_df_2016, shape_2016 = extract_features_for_classification(eopatch_2016)
    
    # Make predictions (these will be integer class indices)
    print("Making predictions...")
    pred_2024 = predict_and_reshape(rf_model, features_df_2024, shape_2024)
    pred_2016 = predict_and_reshape(rf_model, features_df_2016, shape_2016)
    
    # Create change detection map
    change_map = (pred_2024 != pred_2016).astype(int)
    
    # Visualize results
    print("Visualizing results...")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot original images
    # Get RGB bands (B04, B03, B02) for visualization
    rgb_2024 = np.zeros((*shape_2024, 3))
    rgb_2024[:, :, 0] = eopatch_2024.data['BANDS'][0][:, :, 3]  # B04 (Red)
    rgb_2024[:, :, 1] = eopatch_2024.data['BANDS'][0][:, :, 2]  # B03 (Green)
    rgb_2024[:, :, 2] = eopatch_2024.data['BANDS'][0][:, :, 1]  # B02 (Blue)
    
    rgb_2016 = np.zeros((*shape_2016, 3))
    rgb_2016[:, :, 0] = eopatch_2016.data['BANDS'][0][:, :, 3]  # B04 (Red)
    rgb_2016[:, :, 1] = eopatch_2016.data['BANDS'][0][:, :, 2]  # B03 (Green)
    rgb_2016[:, :, 2] = eopatch_2016.data['BANDS'][0][:, :, 1]  # B02 (Blue)
    
    # Enhance contrast for better visualization
    rgb_2024 = np.clip(rgb_2024 * 3.5, 0, 1)
    rgb_2016 = np.clip(rgb_2016 * 3.5, 0, 1)
    
    axs[0, 0].imshow(rgb_2024)
    axs[0, 0].set_title('Original Image (20.8.2024)')
    axs[0, 0].axis('off')
    
    axs[1, 0].imshow(rgb_2016)
    axs[1, 0].set_title('Original Image (27.8.2016)')
    axs[1, 0].axis('off')
    
    # Create discrete colors for classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Plot prediction maps
    im1 = axs[0, 1].imshow(pred_2024, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axs[0, 1].set_title('Predicted Classes (20.8.2024)')
    axs[0, 1].axis('off')
    
    im2 = axs[1, 1].imshow(pred_2016, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axs[1, 1].set_title('Predicted Classes (27.8.2016)')
    axs[1, 1].axis('off')
    
    # Plot change detection map in bottom right with black and white colormap
    bw_cmap = plt.matplotlib.colors.ListedColormap(['black', 'white'])
    im3 = axs[1, 2].imshow(change_map, cmap=bw_cmap, vmin=0, vmax=1)
    axs[1, 2].set_title('Change Detection Map')
    axs[1, 2].axis('off')
    
    # Add a custom legend for class colors in the top right
    axs[0, 2].axis('off')
    
    # Create a figure for the combined legend
    legend_elements = []
    for i, class_name in enumerate(class_names):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=colors[i], label=class_name))
    
    # Add change detection legend elements
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, color='black', label='No Change'))
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, color='white', label='Change'))
    
    # Create two separate legends
    # First legend for land cover classes
    land_cover_elements = legend_elements[:-2]  # All except the last two elements
    land_legend = axs[0, 2].legend(handles=land_cover_elements, loc='upper center', 
                                  fontsize=12, title="Land Cover Classes", 
                                  title_fontsize=14, frameon=True)
    axs[0, 2].add_artist(land_legend)
    
    # Second legend for change detection
    change_elements = legend_elements[-2:]  # Last two elements
    axs[0, 2].legend(handles=change_elements, loc='lower center', 
                    fontsize=12, title="Change Detection", 
                    title_fontsize=14, frameon=True)
    
    axs[0, 2].set_title('Legend')
    
    
    plt.tight_layout()
    plt.savefig('rf_classification_results.png')
    plt.show()
    
    print("Script completed successfully!") 