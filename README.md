# 4DSTEMorientationMapping

A package designed to train a model for identifying orientations of a single crystal from 4-dimensional scanning transmission electron microscopy diffraction patterns.


We have made modifications and additions to the original py4DSTEM codebase to support simulations of dynamic diffraction from SO3 proper orientation matrix and to support calculation of sparse correlation value between two Bragg vector point list.



## License Notice
- Licensed under the GNU General Public License v3.0
- The original py4DSTEM project is available at: https://github.com/py4dstem/py4DSTEM

## Crystal Structures Used
The unit cell CIF files used in this study are based on entries from the Inorganic Crystal Structure Database (ICSD) (https://icsd.products.fiz-karlsruhe.de/) database:

- ICSD #136042: Cu   (cubic)       'F m -3 m'     crystal name used in codes 'Cu_fcc'
- ICSD #63281:  Cu2O (cubic)       'P n -3 m Z'   crystal name used in codes 'Cu_cubic'
- ICSD #67850:  CuO  (monoclinic)  'C 1 2/c 1'    crystal name used in codes 'CuO_monoclinic'

Due to licensing restrictions, CIF files are not included. Users with access to the ICSD can retrieve the data via these entry numbers.
If you don't have access to the ICSD, you can get unit cell information and cif files from other open-source projects including
materials project (https://next-gen.materialsproject.org/).


## How to generate synthetic data
#### To generate synethetic training data, one should first sample thickness and orientations from given crystal unit cell.

```bash
./scripts/data_gen_01_synthetic_training_data/step_01_sample_orientations_and_thickness/sample_orientation_and_thickness.sh




## Acknowledgments
This project is supported by the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, a program of Schmidt Sciences, LLC
