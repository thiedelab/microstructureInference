# 4DSTEMorientationMapping

A package designed to train a model for identifying orientations of a single crystal from 4-dimensional scanning transmission electron microscopy (4D-STEM) diffraction patterns.


We have made modifications and additions to the original py4DSTEM codebase to support simulations of dynamic diffraction from SO3 proper orientation matrix and to support calculation of sparse correlation value between two Bragg vector point list.


## License Notice
- Licensed under the GNU General Public License v3.0
- Modified version of py4DSTEM codes are in third_party directory
- The original py4DSTEM project is available at: https://github.com/py4dstem/py4DSTEM


### Crystal Structures Used
The unit cell cif files used in this study are based on entries from the Inorganic Crystal Structure Database (ICSD) (https://icsd.products.fiz-karlsruhe.de/) database:

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

#### How to generate synthetic training data for training neural network models

###### step 01. We first sample thickness and orientations from given crystal unit cell.

```bash
./scripts/data_generate_01_synthetic_training_data/step_01_sample_orientations_and_thickness/sample_orientation_and_thickness.sh
```

###### step 02. From the sampled orientations and thickness, we simulate dynamic diffraction patterns and save them in table format.

```bash
./scripts/data_generate_01_synthetic_training_data/step_02_simulate_dynamic_diffraction_patterns/simulate_dynamic_diffraction.sh
```


###### step 03. For each diffraction pattern, we further digitize Bragg disk positions and intensities (still in table format)

```bash
./scripts/data_generate_01_synthetic_training_data/step_03_digitize_bins_for_Bragg_disk_positions_and_intensities/digitize_dynamic_diffraction.sh
```

###### step 04. Finally, we merge all data and split it into training data and validation data

```bash
./scripts/data_generate_01_synthetic_training_data/step_04_merge_all_data_and_split_into_training_and_validation/merge_and_split.sh
```


#### How to generate synthetic 4D-STEM data


#### How to extract Bragg Disks from experimental 4D-STEM data

#### How to map experimental diffraction patterns to corresponding orientations



## Acknowledgments
This project is supported by the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, a program of Schmidt Sciences, LLC
