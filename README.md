<!-- @format -->

<img src="./assets/header.png"/>

<p align="center">
  <img src="./assets/shared image.png" alt="Header" width="200"/>
</p>

<h1 align="center">
Tiamat 
  
Electron Orbital Shift in SIRIUS
</h1>

<p align="center">
  <i>Exploring Machine Learning models to predict earthquake interaction in Sirius</i><br/>
  Ilum - School of Science | Campinas, 2026
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
  <img src="https://img.shields.io/badge/domain-Signal%20Processing-green">
  <img src="https://img.shields.io/badge/focus-Machine%20Learning-blueviolet">
</p>

---

## Introduction

This project aims to investigate a possible correlation between shifts in the electron orbit at the SIRIUS particle accelerator and earthquakes around the world. It was developed as an undergraduate research project and submitted as part of the Tiamat Project at ILUM School of Science under the supervision of Dr. Daniel Cassar.

## How to execute?

1. Install and extract the archive in a preferred folder.

2. Open the folder with the program of choice. The autor recommends creating a virtual enviroment for the next step.

3. run, in the terminal, the following command:

``pip install -r requirements.txt``

4. Go to ``Scripts`` and run the files in this order:
- archiver_api.py
- data_treatment.py
- seismo_detection.py

## Contents:

<ul>
  <li> :file_folder: <b>Project</b>: contains the scripts, files, and data used in the project.</li>
    <ul>
      <li> :file_folder: <b>Data</b>: stores the data used throughout the project.</li>
        <ul>
          <li> :file_folder: <b>log_data</b>: contains data and tools related to data logging.</li>
            <ul>
              <li> :card_file_box: <b>olog_data.csv</b>: file containing the collected data.</li>
              <li> :page_facing_up: <b>olog_data_retriever.ipynb</b>: notebook used for data retrieval and processing.</li>
            </ul>
        </ul>
      <li> :file_folder: <b>GDM52_program</b>: program responsible for processing and analysis related to the GDM52.</li>
      <li> :file_folder: <b>Scripts</b>: contains scripts related to data acquisition, processing, and analysis.</li>
        <ul>
          <li> :page_facing_up: <b>api.ipynb</b>: notebook related to API usage.</li>
          <li> :page_facing_up: <b>archiver_api.py</b>: functions for communicating with the Archiver API.</li>
          <li> :page_facing_up: <b>arrival_time.py</b>: calculation and processing of seismic wave arrival times.</li>
          <li> :page_facing_up: <b>association_v2.py</b>: temporal association analysis between seismic events and the recorded data.</li>
          <li> :page_facing_up: <b>constants.py</b>: collection of constants used by the other scripts.</li>
          <li> :page_facing_up: <b>data_treatment.py</b>: functions for data processing and preparation.</li>
          <li> :page_facing_up: <b>feature_analysis.py</b>: analysis of the features used in the project.</li>
        </ul>
      <li> :page_facing_up: <b>data_treatment.py</b>: script for data processing and preparation.</li>
      <li> :page_facing_up: <b>feature_analysis.py</b>: script for feature analysis.</li>
      <li> :page_facing_up: <b>final_models.py</b>: contains the final models used in the project.</li>
      <li> :page_facing_up: <b>model_searches.py</b>: performs hyperparameter search and model evaluation.</li>
      <li> :page_facing_up: <b>optuna_searches.py</b>: performs hyperparameter optimization using Optuna.</li>
      <li> :page_facing_up: <b>seismo_detection.py</b>: script related to seismic event detection and classification.</li>
      <li> :file_folder: <b>assets</b>: contains image files used throughout the project.</li>
        <ul>
          <li> :framed_picture:  <b>Rodape_eleição.png</b>: image used in the project.</li>
          <li> :framed_picture: <b>header.png</b>: image used in the header.</li>
          <li> :framed_picture: <b>shared image.png</b>: shared image used in the project.</li>
        </ul>
      <li> :page_facing_up: <b>LICENSE</b>: project usage and distribution license.</li>
      <li> :page_facing_up: <b>README.md</b>: main project documentation.</li>
      <li> :page_facing_up: <b>Project Report - Tiamat - Vinícius Mariano...</b>: project report.</li>
      <li> :spiral_notepad: <b>requirements.txt</b>: file containing the libraries required to run the project.</li>
    </ul>
</ul>

## Relevant xkcd


![img](https://imgs.xkcd.com/comics/earthquake_early_warnings.png)

Source: https://xkcd.com/2219/

---

# Teacher

<table>
  <td align="center">
      <a href="https://github.com/drcassar">
        <img src="https://avatars.githubusercontent.com/u/9871905?v=4" width="100px;" alt="Foto do Cassar no Github"/><br>
        <b>Prof. Dr. Daniel R. Cassar</b>
      </a>
    </td>
</table>

---

# Authors

<table>
  <td align="center">
      <a href="https://github.com/ViniCutolo">
        <img src="https://avatars.githubusercontent.com/u/271505502?v=4" width="100px;" alt="Photo of Vinícius" style="border-radius:100%;"/><br>
        <b>Vinícius Cutolo</b>
      </a>
    </td>
  <td>
    <td align="center">
      <a href="https://github.com/FipaLipe">
        <img src="https://avatars.githubusercontent.com/u/90717547?v=4" width="100px;" alt="Photo of Filipi" style="border-radius:100%;"/><br>
        <b>Filipi Martins</b>
      </a>
    </td>
</table>
