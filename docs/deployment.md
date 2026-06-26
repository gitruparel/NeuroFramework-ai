# Deployment Details

## Backend (FastAPI)
The backend containerizes endpoints. It is optimized to parse requests asynchronously. High computational tasks (like preprocessing and 3D CNN execution) delegate work to background tasks or queue workers.

## Frontend (Streamlit)
A research dashboard interface that communicates with the backend API to visualize loaded MRI slices (axial, coronal, sagittal), display structural classification or segmentation maps, and download generated reports.
