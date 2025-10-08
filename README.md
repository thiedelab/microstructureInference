# 4DSTEMorientationMapping

A package designed to train a model for identifying orientations of a single crystal from 4-dimensional scanning transmission electron microscopy (4D-STEM) diffraction patterns.


We have made modifications and additions to the original py4DSTEM codebase to support simulations of dynamic diffraction from SO3 proper orientation matrix and to support calculation of sparse correlation value between two Bragg vector point list.


## License Notice
- Licensed under the GNU General Public License v3.0
- Modified version of py4DSTEM codes are in third_party directory
- The original py4DSTEM project is available at: https://github.com/py4dstem/py4DSTEM

---

### Crystal Structures Used
The unit cell cif files used in this study are based on entries from the Inorganic Crystal Structure Database (ICSD) (https://icsd.products.fiz-karlsruhe.de/):

| ICSD collection code | Chemical formula | crystal space group | crystal space group IT number | variable string used in codes |
|:--------------------:|:----------------:|:-------------------:|:-----------------------------:|:-----------------------------:|
| ICSD #136042         |    Cu            |     F m -3 m        |            225                |           'Cu_fcc'            |
| ICSD #63281          | Cu<sub>2</sub>O  |     P n -3 m Z      |            224                |           'Cu2O_cubic'        |
| ICSD #67850          |   CuO            |     c 1 2/c 1       |            15                 |           'CuO_monoclinic'    |


Due to licensing restrictions, cif files are not included. Users with access to the ICSD can retrieve the data via these entry numbers.
If you don't have access to the ICSD, you can get unit cell information and cif files from other open-source projects including
materials project (https://next-gen.materialsproject.org/).


### Descriptions for training neural networks models

### Descriptions of scripts for data generation and data analysis

---

#### How to generate synthetic training data for training neural network models

###### step 01. We first sample thickness and orientations from given crystal unit cell.
###### step 02. From the sampled orientations and thickness, we simulate dynamic diffraction patterns and save them in table format.
###### step 03. For each diffraction pattern, we further digitize Bragg disk positions and intensities (still in table format)
###### step 04. Finally, we merge all data and split it into training data and validation data
###### please make sure to make bash files in scripts directory executable by chmod +x 
```bash
./scripts/data_generate_01_synthetic_training_data/DATA_GENERATE_01_generate_synthetic_training_data.sh
```

#### How to generate synthetic 4D-STEM data
###### step 01. We first sample background signals from experimental 4D-STEM data and make synthetic scanspace filled with crystal grains.
###### step 02. Thereafter, we randomly sample orienations
###### step 03,04. Using orientations and sampled background signals, we simulated diffraction patterns and assign it to each scan space pixel. step 03 generate synthetic 4DSTEM data for single Cu fcc crystal. step 04 generate synthetic 4DSTEM data for 3 crystals; Cu fcc, Cu2O cubic, CuO monoclinic crystals.
###### please make sure to make bash files in scripts directory executable by chmod +x 
```bash
./scripts/data_generate_02_synthetic_4DSTEM_data/DATA_GENERATE_02_generate_synthetic_4DSTEM_data.sh
```

#### How to map a diffraction pattern of experimental 4D-STEM data to a table of detected Bragg disks
###### In this proejct, we identify orienation of single crystal from a diffraction pattern by using a list of Bragg disks in the diffraction pattern; we map each diffraction pattern to a list (or table) of Bragg disks.
###### The map is obtained by detecting Bragg disks in a diffraction pattern using correlative template matching
###### The correlation template is obtained by sampling direct beam from diffraction pattern and averaging them
###### We gently note that we perform difference of gaussian preprocessing prior to Bragg disk detection to remove backgrounds.
###### please make sure to make bash files in scripts directory executable by chmod +x 
```bash
./scripts/data_analyses_01_mapping_diffractionPattern_to_BraggDiskTable/DATA_ANALYSES_01_map_diffPatt_to_BraggDiskTable.sh
```

#### How to predict orientations from experimental 4D-STEM diffraction patterns



## Acknowledgments
This project is supported by the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, a program of Schmidt Sciences, LLC
