# Satellite Image Time Series Analysis with Python

Materials for the [*Satellite Image Time Series Analysis with Python*](https://s34iclassroom.fgg.uni-lj.si/course/view.php?id=5).

Prepared by:  
Krištof Oštir, Bujar Fetai, Matej Račič, Tanja Grabrijan (University of Ljubljana)

## Preparation

The repository consists of installation instructions, notebooks and practical information.


The repository can be synchronized using `git pull` or downloaded as a zip file. The data used for the practical exercise can be downloaded from [s34iclassroom](https://s34iclassroom.fgg.uni-lj.si/course/view.php?id=5). 

The practical has installation instructions which should be completed in advance.

## Installation instructions
We will be using [Anaconda](https://www.anaconda.com/), which can be installed from the [website](https://www.anaconda.com/products/distribution#Downloads). If you have limited resources we suggest the us of [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Once installed open Anaconda Prompt and move to the location of the extracted repository `cd Downloads\S34I-training\`. If you have downloaded it to a different drive change the location to the drive first, example `D:` and press Enter.

Here you can create a new environment for this tutorial using the provided environment.yml file:

```
conda update -n base -c defaults conda
conda env create --file environment.yml
conda activate eo
```

Alternatively, you can use pip to install the libraries using 'pip' and follow the tutorial. This will take some time. Once installed run `jupyter lab` and a browser tab will open.

## Practicals
We will be using the notebooks available in the corresponding folders. To run the notebook you will need a [Copernicus Data Space Ecosystem](https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth?client_id=cdse-public&response_type=code&scope=openid&redirect_uri=https%3A//dataspace.copernicus.eu/account/confirmed/1) account, available for free. Once registered you can follow the [slides](https://s34iclassroom.fgg.uni-lj.si/pluginfile.php/92/mod_resource/content/4/Instructions%20Satellite%20Image%20Time%20Series%20Analysis%20with%20Python.pdf) to configure access to the services.

## Additional resources
This tutorial is based on the [materials](https://github.com/sentinel-hub/eo-learn-workshop/) provided by Sinergise. Where you can find even more examples and resources for the [eo-learn](https://github.com/sentinel-hub/eo-learn) library.

## Acknowledgment

Preparation of the materials was financed by the [S34I](https://s34i.eu/) project.

## License
This project is licensed under the terms of the [Apache License](LICENSE).

© Copyright 2024 University of Ljubljana, Faculty of Civil and Geodetic Engineering
