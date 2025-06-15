#!/bin/bash
cd /data/Hidden_Lineage/web
# Activate conda environment
source /home/nagy/miniconda3/bin/activate hl_pipeline
streamlit run personal_dashboard.py --server.port 8502 --server.headless true --server.address 0.0.0.0
