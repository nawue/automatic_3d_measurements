# Obtención de medidas antropométricas a partir de modelos 3D de cuerpos humanos

Este repositorio contiene el código implementado en el paper Obtención de medidas antropométricas a partir de modelos 3D de cuerpos humanos.

## Instalación

Check out and install this repository:
```
$ git clone [automatic_3d_measurements](https://github.com/nawue/automatic_3d_measurements.git)
$ pip install trymesh open3d numpy click pandas datetime Pillow==9.0.0

$ git clone https://github.com/sergeyprokudin/bps.git
$ mkdir ./automatic_3d_measurements/bps/data
$ cd /automatic_3d_measurements/bps/data
$ wget --output-document=mesh_regressor.h5 https://www.dropbox.com/s/u3d1uighrtcprh2/mesh_regressor.h5?dl=0
$ mv /automatic_3d_measurements/mesh_regressor.h5 /content/bps/data/mesh_regressor.h5
$ cp /automatic_3d_measurements/content/bps/bps_demos/smpl_mesh_faces.txt /content/smpl_mesh_faces.txt

$ python auto_measurement.py --help
```
