## SUVEL - Tracking the photospheric horizontal velocity field with shallow UNet models
**If you use any part of the code in this repository, please cite the following work**

**Liu et al. 2024, Tracking the Photospheric Horizontal Velocity Field with Shallow U-Net Models**

**Contact me at jiajialiu@ustc.edu.cn if you have any question or suggestions**

**Use SUVEL on your own risks** 

### Notes
* SUVEL takes three consecutive observations of the photospheric intensity in continuum or/and three consecutive observations of the photospheric vertical/LOS magnetic field strength as its input.
* The following instructions take a typical Linux system as an example, it is similar to use SUVEL in Windows or MacOS
* One must have a suitable GPU to run SUVEL properly. Although Tensorflow can be run at CPU, it will be very slow.

### Instructions (Run the demo)

* Install Anaconda (https://www.anaconda.com/download)
* Clone or download this repository
* Open your terminal and go to the local respository directory by typing in (suppose the directory is localted at ./SUVEL)
    ```
    cd ./SUVEL
    ```
* Create conda environment and install dependencies
    ```
    conda env create
    ```
* The above commend will create a conda environment named as *suvel*. Activate it by typing in
    ```
    source activate suvel
    ```
* Run the demo file and check the results
    ```
    python demo.py
    ```

### Write your own script based on demo.py
**Minimum codes required in your script to use suvel**

* **NOTE: SUVEL CANNOT BE USED ON ANY OTHER OBSERVATIONS**, it can only be used on **PHOTOSPHERIC INTENSITY AT CONTINUUM and/or PHOTOSPHERIC VERTICAL/LOS MAGNETIC FIELD STRENGTH**. SUVEL has only been tested on QUIET-SUN regions.

* load your own data, suppose they are named as intensity and magnetic. They must be in shape of [ny, nx, nt], where ny is the number of pixels along the y axis, nx is the number of pixels along the x axis, and nt must be 3. The last dimension has 3 elements, they are the intensity at t-dt, t, and t+dt. dt is the cadence of the data.

* Normalize intensity and magnetic to [0, 1]. The upper and lower limits for normalization depends on your data. A good practice is to normalize the magnetic field to [0, 1] with -200 to 200.

* Define the model path *suvel_path*

* Do prediction using the following codes
    ```
    # do prediction using intensity only
    vi = suvel(intensity=intensity, model_path=suvel_path)

    # do prediction using magnetic field only
    vm = suvel(magnetic=magnetic, model_path=suvel_path)

    # do prediction using both
    vh = suvel(intensity=intensity, magnetic=magnetic, model_path=suvel_path)
    ```

### Outputs of the demo
**Comparison between the ground-truth velocity field (top) and reconstructed velocity field by the intensity model (bottom)**
![Comparison between the ground-truth velocity field (top) and reconstructed velocity field by the intensity model (bottom)](https://github.com/PyDL/SUVEL/blob/main/fig_intensity_model.png)

**Comparison between the ground-truth velocity field (top) and reconstructed velocity field by the magnetic model (bottom)**
![Comparison between the ground-truth velocity field (top) and reconstructed velocity field by the intensity model (bottom)](https://github.com/PyDL/SUVEL/blob/main/fig_magnetic_model.png)

**Comparison between the ground-truth velocity field (top) and reconstructed velocity field by the hybrid model (bottom)**
![Comparison between the ground-truth velocity field (top) and reconstructed velocity field by the intensity model (bottom)](https://github.com/PyDL/SUVEL/blob/main/fig_hybrid_model.png)
