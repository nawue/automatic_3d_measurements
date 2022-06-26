import os
import sys
import math
import click
import tests
import trimesh
import numpy as np
import pandas as pd
import open3d as o3d

from os import listdir
from datetime import datetime
from typing import Any, Union
from subprocess import check_output
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.append("/automatic_3d_measurements/bps/bps_demos/")
import run_alignment

class Measures:
    def __init__(self):
        self.name = ""
        self.npoints = 0
        self.points = []

def register_pointclounds(directory):
    """
    This function will read all PointClouds obtained from different points of view
    PointClounds whitin the directory must be named e.g. pointcloud1-CloudTransfRGB.ply
    """
    ply_combined = o3d.geometry.PointCloud()
    voxel_size = 0.02
    # plys = [_ for _ in listdir(directory) if "CloudTransfRGB" in _]
    plys = [_ for _ in listdir(directory) if "CloudTransf-debug" in _]
    if len(plys) == 0:
        print("No PointClouds")
        exit()

    for ply_id in plys:
        ply_combined += o3d.io.read_point_cloud(directory + ply_id)

    result = ply_combined.voxel_down_sample(voxel_size=voxel_size)
    # Recompute the normal of the downsampled point cloud
    result.estimate_normals()

    return result

class DeletePointsBase:
    """
    This class removes points below the 3D body position using a plane
    """

    def __init__(self, body, y_position):
        # 3D body points and normals
        self.body = body
        X1 = np.asarray(self.body.points)[:, 0]
        X2 = np.asarray(self.body.points)[:, 1]
        X3 = np.asarray(self.body.points)[:, 2]
        N1 = np.asarray(self.body.normals)[:, 0]
        N2 = np.asarray(self.body.normals)[:, 1]
        N3 = np.asarray(self.body.normals)[:, 2]

        self.Xor = np.stack((X1[::1], X2[::1], X3[::1]), axis=1)
        self.NXor = np.stack((N1[::1], N2[::1], N3[::1]), axis=1)

        self.y_position = y_position

        # Corners and normal of the plane
        self.pplane = np.zeros((4, 3))
        self.nplane = np.zeros((4, 3))

        self.pplane[0, :] = np.array([-1.321164, -0.500000, 1.009144])
        self.pplane[1, :] = np.array([-1.321164, 0.500000, 1.009144])
        self.pplane[2, :] = np.array([-0.979144, -0.500000, 1.948836])
        self.pplane[3, :] = np.array([-0.979144, 0.500000, 1.948836])

        self.nplane[0, :] = np.array([0.939693, 0.000000, -0.342020])
        self.nplane[1, :] = np.array([0.939693, 0.000000, -0.342020])
        self.nplane[2, :] = np.array([0.939693, 0.000000, -0.342020])
        self.nplane[3, :] = np.array([0.939693, 0.000000, -0.342020])

    def rotation_matrix_from_vectors(self, vec1, vec2):
        a, b = (
            (vec1 / np.linalg.norm(vec1)).reshape(3),
            (vec2 / np.linalg.norm(vec2)).reshape(3),
        )
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def AlignZandDeletePointsBase(self):
        # Align with Z=0 plane
        rot = self.rotation_matrix_from_vectors(self.nplane[0, :], [1, 0, 0])
        self.Xor = (rot @ self.Xor.T).T
        pplaneRot = (rot @ self.pplane[0, :].T).T
        pplaneRot[0] = pplaneRot[0] + self.y_position * 0.05
        desp = [pplaneRot[0], 0, 0]
        self.Xor = self.Xor - desp

        # Delete Z < 0
        idxBody: Union[bool, Any] = self.Xor[:, 0] >= 0.1
        idxFeet = (self.Xor[:, 0] > 0) & (self.Xor[:, 0] < 0.1)

        self.Xor = np.concatenate((self.Xor[idxBody, :], self.Xor[idxFeet, :]))
        self.NXor = np.concatenate((self.NXor[idxBody, :], self.NXor[idxFeet, :]))

        # Pass xyz to Open3D.o3d.geometry.PointCloud
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(self.Xor)
        result.normals = o3d.utility.Vector3dVector(self.NXor)

        # return trimesh.Trimesh(vertices=self.Xor, vertex_normal=self.NXor)
        return result

def alignmentWithBPS(body, dir_model_align):
    # estimate radius for rolling ball
    distances = body.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        body, o3d.utility.DoubleVector([radius, radius * 2])
    )

    # create the triangular mesh with the vertices and faces from open3d
    mesh_result = trimesh.Trimesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals),
    )

    model_align = trimesh.load(dir_model_align, process=False)
    align_matrix = trimesh.registration.mesh_other(
        mesh_result, model_align, scale=False
    )

    result = trimesh.Trimesh(vertices=body.points, vertex_normals=body.normals)
    result.apply_transform(align_matrix[0])

    return result

def aligmentBodies(dir_model_align, dir_model, output_path):
    mesh_result = trimesh.load(dir_model, process=False)
    model_align = trimesh.load(dir_model_align, process=False)
    align_matrix = trimesh.registration.mesh_other(
        mesh_result, model_align, scale=False
    )
    mesh_result.apply_transform(align_matrix[0])
    mesh_result.export(output_path, file_type="obj")

def get_ids_measurements(area_measurement, output_bps):
    """
    How to obtain vertex ids to make measurements?
    1. We use BPS models to create PLYs with the same vertex id order
    2. Using one BPS, we create different PLSs associates with the different measurement zones (hip, wrist, etc)
    3. We read each XYZ position on each PLY created on the previous step
    4. We search each XYZ on the original BPS model using a for-loop, when its found we got the vertex id.

    How to obtain measurement zones each time we compute a BPS model?
    1. Since we have the vertex ids of all zones, we only have to search these ids on the generated BPS model and get the XYZ positions
    """

    onlyfiles2 = [_ for _ in listdir(area_measurement)]
    original = trimesh.load(output_bps, process=False)
    vertex_ids = []
    for _ in onlyfiles2:
        parte = trimesh.load(area_measurement + _, process=False)
        vertex_ids.append(np.zeros(len(parte.vertices)))

    for _ in range(0, len(onlyfiles2)):
        parte = trimesh.load(area_measurement + onlyfiles2[_], process=False)
        # print(onlyfiles2[_][:-4])
        for ids1 in range(0, len(parte.vertices)):
            for ids2 in range(0, len(original.vertices)):
                if (
                    round(parte.vertices[ids1][0], 4)
                    == round(original.vertices[ids2][0], 4)
                    and round(parte.vertices[ids1][1], 4)
                    == round(original.vertices[ids2][1], 4)
                    and round(parte.vertices[ids1][2], 4)
                    == round(original.vertices[ids2][2], 4)
                ):
                    vertex_ids[_][ids1] = ids2

    np.save("vertex_ids.npy", vertex_ids)
    return vertex_ids

def get_XYZ_areas(vertex_ids, area_measurement, file_measurements, output_bps):
    onlyfiles2 = [_ for _ in listdir(area_measurement)]
    original = trimesh.load(output_bps, process=False)

    """
    open(file_measurements, "w").close()  # erase file
    with open(file_measurements, "w") as f:
        for _ in range(0, len(onlyfiles2)):
            f.write(onlyfiles2[_][:-4])
            f.write("\n")
            f.write(str(len(vertex_ids[_])))
            f.write("\n")
            for __ in range(0, len(vertex_ids[_])):
                f.write(str(original.vertices[int(vertex_ids[_][__])])[1:-2])
                f.write("\n")
    """
    bodyParts = []
    for _ in range(0, len(onlyfiles2)):
        part = Measures()
        part.name = onlyfiles2[_][:-4]
        part.npoints = len(vertex_ids[_])
        part.points = []
        for __ in range(0, len(vertex_ids[_])):
            part.points.append(
                np.asarray(np.round(original.vertices[int(vertex_ids[_][__])], 5))
            )
        bodyParts.append(part)

    return bodyParts

def createCube(bodyPart, parte):
    """
    if parte.name == "wrist" or parte.name == "forearm":
        dx = 0.1
    elif parte.name == "hand":
        dx = 0.1
        dy = 0.1
        dz = 0.1
    else:
        dx = 0.03
    dy = 0.04
    dz = 0.05
    """
    dx, dy, dz = 0.1, 0.1, 0.1
    # Cube corners
    p1 = [
        np.min(bodyPart[:, 0]) - dx,
        np.min(bodyPart[:, 1]) - dy,
        np.min(bodyPart[:, 2]) - dz,
    ]
    p2 = [
        np.min(bodyPart[:, 0]) - dx,
        np.max(bodyPart[:, 1]) + dy,
        np.min(bodyPart[:, 2]) - dz,
    ]
    p3 = [
        np.min(bodyPart[:, 0]) - dx,
        np.min(bodyPart[:, 1]) - dy,
        np.max(bodyPart[:, 2]) + dz,
    ]
    p4 = [
        np.min(bodyPart[:, 0]) - dx,
        np.max(bodyPart[:, 1]) + dy,
        np.max(bodyPart[:, 2]) + dz,
    ]
    p5 = [
        np.max(bodyPart[:, 0]) + dx,
        np.min(bodyPart[:, 1]) - dy,
        np.min(bodyPart[:, 2]) - dz,
    ]
    p6 = [
        np.max(bodyPart[:, 0]) + dx,
        np.max(bodyPart[:, 1]) + dy,
        np.min(bodyPart[:, 2]) - dz,
    ]
    p7 = [
        np.max(bodyPart[:, 0]) + dx,
        np.min(bodyPart[:, 1]) - dy,
        np.max(bodyPart[:, 2]) + dz,
    ]
    p8 = [
        np.max(bodyPart[:, 0]) + dx,
        np.max(bodyPart[:, 1]) + dy,
        np.max(bodyPart[:, 2]) + dz,
    ]

    cube = np.array([p1, p2, p3, p4, p5, p6, p7, p8])

    return cube

def plane_from_points(points):

    centroid = np.mean(points, axis=0)

    # Calc full 3x3 covariance matrix, excluding symmetries:
    xx, xy, xz = 0.0, 0.0, 0.0
    yy, yz, zz = 0.0, 0.0, 0.0

    for p in points:
        r = p - centroid
        xx += np.dot(r[0], r[0])
        xy += np.dot(r[0], r[1])
        xz += np.dot(r[0], r[2])
        yy += np.dot(r[1], r[1])
        yz += np.dot(r[1], r[2])
        zz += np.dot(r[2], r[2])

    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy

    det_max = np.max([det_x, det_y, det_z])
    if det_x > det_y:
        det_max = det_x
    else:
        det_max = det_y

    if det_z > det_max:
        det_max = det_z

    if det_max <= 0.0:
        return False

    # Pick path with best conditioning:
    if det_max == det_x:
        x = (det_x,)
        y = xz * yz - xy * zz
        z = xy * yz - xz * yy
    elif det_max == det_y:
        x = xz * yz - xy * zz
        y = det_y
        z = xy * xz - yz * xx

    else:
        x = xy * yz - xz * yy
        y = xy * xz - yz * xx
        z = det_z

    dir = [x, y, z]
    normal_x, normal_y, normal_z = 0, 0, 0
    if x != 0:
        if type(x) is tuple:
            normal_x = x[0] / len(points)
        else:
            normal_x = x / len(points)
    if y != 0:
        normal_y = y / len(points)
    if z != 0:
        normal_z = z / len(points)

    return centroid, [normal_x, normal_y, normal_z]

def calculatePlanes(cube):
    planes = []
    centroid, normal = plane_from_points(cube[0:4])
    planes.append([centroid, normal])

    centroid, normal = plane_from_points(cube[4:8])
    planes.append([centroid, normal])

    aux = [0, 2, 4, 6]
    centroid, normal = plane_from_points(cube[aux])
    planes.append([centroid, normal])

    aux = [1, 5, 3, 7]
    centroid, normal = plane_from_points(cube[aux])
    planes.append([centroid, normal])

    aux = [2, 6, 3, 7]
    centroid, normal = plane_from_points(cube[aux])
    planes.append([centroid, normal])

    aux = [0, 1, 4, 5]
    centroid, normal = plane_from_points(cube[aux])
    planes.append([centroid, normal])

    return planes

def sliceBody(planes, output_path, parte):
    
    model = trimesh.load(output_path, process=False)
    """
    faces = []
    with open("/home/nawue/Desktop/Papers/Soco/smpl_mesh_faces.txt") as f:
        for line in f:
            aux = line.split(" ")
            aux = list(filter(lambda a: a != "", aux))
            faces.append([int(aux[0]), int(aux[1]), int(aux[2])])
    bps = trimesh.load(
        "/home/nawue/Desktop/Papers/Soco/tests/multiway_registration_aligned.ply"
    )
    model = trimesh.Trimesh(bps.vertices, faces)
    model.export("bps_texture.obj")
    """
    """
    import polyscope as ps

    ps.init()

    # Read & register the mesh
    vertices = model.vertices.copy()
    faces = model.faces.copy()
    ps_mesh = ps.register_surface_mesh("my mesh", vertices, faces)
    ps_mesh.set_cull_whole_elements(True)

    # Add a slice plane
    ps_plane0 = ps.add_scene_slice_plane()
    ps_plane0.set_pose(planes[0][0], np.dot(planes[0][1],1))
    ps_plane0.set_draw_plane(True)
    ps_plane0.set_draw_widget(True)

    ps_plane1 = ps.add_scene_slice_plane()
    ps_plane1.set_pose(planes[1][0], np.dot(planes[1][1],-1))
    ps_plane1.set_draw_plane(True)
    ps_plane1.set_draw_widget(True)

    ps_plane2 = ps.add_scene_slice_plane()
    ps_plane2.set_pose(planes[2][0], planes[2][1])
    #ps_plane2.set_draw_plane(True)
    #ps_plane2.set_draw_widget(True)

    ps_plane3 = ps.add_scene_slice_plane()
    ps_plane3.set_pose(planes[3][0], np.dot(planes[3][1],-1))
    #ps_plane3.set_draw_plane(True)
    #ps_plane3.set_draw_widget(True)

    ps_plane4 = ps.add_scene_slice_plane()
    ps_plane4.set_pose(planes[4][0], np.dot(planes[4][1],-1))

    ps_plane5 = ps.add_scene_slice_plane()
    ps_plane5.set_pose(planes[5][0], planes[5][1])

    #ps.screenshot(transparent_bg=False)
    ps.show()
    ps.remove_all_structures()
    """

    if parte != "hand" or parte != "wrist":
        aux = model.slice_plane(planes[0][0], planes[0][1])
        if parte != "neck" or parte != "forearm":
            aux = aux.slice_plane(planes[1][0], np.dot(planes[1][1], -1))
            aux = aux.slice_plane(planes[2][0], planes[2][1])
            aux = aux.slice_plane(planes[3][0], np.dot(planes[3][1], -1))
            aux = aux.slice_plane(planes[4][0], np.dot(planes[4][1], -1))
            aux = aux.slice_plane(planes[5][0], planes[5][1])
    else:
        aux = model.copy()
    return aux

def calculate_best_fit_plane(part):

    centroid = np.mean(part.points, axis=0)  # Good

    xx, xy, xz = 0.0, 0.0, 0.0
    yy, yz, zz = 0.0, 0.0, 0.0

    for i in range(0, len(part.points)):
        r = np.asarray(part.points[i]) / centroid
        xx += r[0] * r[0]
        xy += r[0] * r[1]
        xz += r[0] * r[2]
        yy += r[1] * r[1]
        yz += r[1] * r[2]
        zz += r[2] * r[2]

    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy
    det_max = 0.0
    if det_x > det_y:
        det_max = det_x
    else:
        det_max = det_y

    if det_z > det_max:
        det_max = det_z
    if det_max == 0.0 or det_max < 0.0:
        return False

    n = [0.0, 0.0, 0.0]

    if det_max == det_x:
        n = [det_x, xz * yz - xy * zz, xy * yz - xz * yy]
    elif det_max == det_y:
        n = [xz * yz - xy * zz, det_y, xy * xz - yz * xx]
    else:
        n = [xy * yz - xz * yy, xy * xz - yz * xx, det_z]

    normal = n / np.linalg.norm(n)
    return centroid, normal

def planeFit(part):
    points = np.array(part.points)
    #points = np.array(part)
    assert points.shape[1] == 3
    centroid = points.mean(axis=0)
    x = points - centroid[None, :]
    U, S, Vt = np.linalg.svd(x.T @ x)
    normal = U[:, -1]
    return centroid, normal

def cross_section(mesh, plane_origin=[0, 0, 0], plane_normal=[1, 0, 0]):

    slice_ = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

    # transformation matrix for to_planar
    # I don't know why
    to_2D = trimesh.geometry.align_vectors(plane_normal, [0, 0, -1])

    slice_2D, to_3D = slice_.to_planar(to_2D=to_2D)

    return slice_2D, to_3D

def save_data_csv(directory, perimetros, tiempos):
    print(directory)
    paciente = directory.split("/")[-3]
    sesion = directory.split("/")[-2].split("2021")[0]
    if sesion == "":
        sesion = directory.split("/")[-2].split("2022")[0]

    csv_path = "/automatic_3d_measurements/experimentacion.csv"
    if not os.path.isfile(csv_path):
        aux = pd.DataFrame()
        aux.to_csv(csv_path)

    df = pd.read_csv(csv_path, index_col=[0])
    df2 = pd.DataFrame(
        {
            "paciente"      : paciente,
            "sesion"        : sesion,
            "muñeca"        : perimetros[0],
            "cadera"        : perimetros[1],
            "cintura"       : perimetros[2],
            "date"          : datetime.now(),
            "measures_time" : tiempos[-2],
            "total_time"    : tiempos[-1],
        },
        index=[0],
    )
    df = df.append(df2, ignore_index=True)
    df.to_csv("/automatic_3d_measurements/experimentacion.csv")


@click.command()
@click.option(
    "--directory",
    #default="/home/nawue/Desktop/Papers/Sarteco/sesiones/52764803C/520210219142452/",
    default='/home/nawue/Desktop/Papers/Sarteco/nahuel/',
    help="Register PointClounds Directory",
    type=click.STRING,
)
@click.option(
    "--dir_model_align",
    default="Model_to_align.obj",
    help="Model located on BPS position",
    type=click.STRING,
)
@click.option(
    "--output_bps",
    default="/home/nawue/Desktop/Papers/Soco/tests/",
    help="BPS Output",
    type=click.STRING,
)
@click.option(
    "--area_measurement",
    default="/home/nawue/Desktop/Papers/Soco/zonas_medidas/",
    help="Folder where the areas to be measured are stored",
    type=click.STRING,
)
@click.option(
    "--file_measurements",
    default="/home/nawue/Desktop/Papers/Soco/measure_model/models/medidas.txt",
    help="Folder where the XYZ zones to be measured will be stored.",
    type=click.STRING,
)
@click.option(
    "--output_path",
    default="/home/nawue/Desktop/Papers/Soco/measure_model/models/Model.obj",
    help="Aligned model output.",
    type=click.STRING,
)
def main(
    directory,
    dir_model_align,
    output_bps,
    area_measurement,
    file_measurements,
    output_path,
):

    click.secho("Executing Automatic Measuraments Pipeline", fg="bright_yellow", bold=True)
    tiempos = []
    
    click.secho("1. Merge PointClounds", fg="yellow")
    start_time = datetime.now()
    body = register_pointclounds(directory)
    #o3d.io.write_point_cloud("exp1.ply", body)
    tiempos.append(datetime.now() - start_time)
    click.secho("Duration: {} \n".format(datetime.now() - start_time), fg="yellow")

    click.secho("2. Remove Points below feet", fg="yellow")
    start_time = datetime.now()
    deletePoints = DeletePointsBase(body, 0)
    body = deletePoints.AlignZandDeletePointsBase()
    # o3d.io.write_point_cloud("exp2.ply", body)
    tiempos.append(datetime.now() - start_time)
    click.secho("Duration: {} \n".format(datetime.now() - start_time), fg="yellow")
    
    click.secho("3. Alignment with the position of the BPS model", fg="yellow")
    start_time = datetime.now()
    body = o3d.io.read_point_cloud("/automatic_3d_measurements/Model.ply")
    body = alignmentWithBPS(body, dir_model_align)
    body.export("exp3.ply")
    tiempos.append(datetime.now() - start_time)
    click.secho("Duration: {} \n".format(datetime.now() - start_time), fg="yellow")
    
    click.secho("4. Generate BPS Model", fg="yellow")
    start_time = datetime.now()
    run_alignment.main("/automatic_3d_measurements/Model.ply", output_bps)
    tiempos.append(datetime.now() - start_time)
    click.secho("Duration: {} \n".format(datetime.now() - start_time), fg="yellow")
    
    click.secho("5. Calculate measurements", fg="yellow")
    start_time = datetime.now()
    #vertex_ids = get_ids_measurements(area_measurement, output_bps + "measures_template.ply")
    vertex_ids = np.load("vertex_ids.npy", allow_pickle=True)
    bodyParts = get_XYZ_areas(
        vertex_ids,
        area_measurement,
        file_measurements,
        output_bps + "multiway_registration_aligned.ply",
    )
    tiempos.append(datetime.now() - start_time)
    click.secho("Duration: {} \n".format(datetime.now() - start_time), fg="yellow")
    
    click.secho("6. Original Model alignment with the position of the BPS model", fg="yellow")
    start_time = datetime.now()
    aligmentBodies(dir_model_align, directory + "Model.obj", output_path)
    tiempos.append(datetime.now() - start_time)
    click.secho("Duration: {} \n".format(datetime.now() - start_time), fg="yellow")
    
    click.secho("7. Calculate planes to slice Model and make better measurement", fg="yellow")
    start_time = datetime.now()
    # onlyfiles2 = [_ for _ in listdir(area_measurement)]

    # slicedPart.export("sliced.obj")
    # slicedPart = trimesh.load('box.obj')
    # slicedPart = trimesh.load('slice.obj')
    # https://github.com/hjwdzh/ManifoldPlus
    # slicedPart = trimesh.load('sliced_watertight.obj')
    # model = trimesh.load("./measure_model/models/Model.obj", process=False)

    perimetros = []
    for _ in bodyParts:
        #click.secho(_.name, fg="bright_yellow", bold=True)
        # print(area_measurement + _.name + ".ply")
        bodyPart = trimesh.load(area_measurement + _.name + ".ply", process=False)
        cube = createCube(bodyPart, _)
        planes = calculatePlanes(cube)
        slicedPart = sliceBody(planes, output_path, _.name)
        #slicedPart.export("sliced.obj")
        # print(slicedPart)

        #centroid, normal = calculate_best_fit_plane(_)
        #tests.main("sliced.obj", centroid, normal)

        #centroid, normal = planeFit(slicedPart.vertices)
        centroid, normal = planeFit(_)
        #tests.main("sliced.obj", centroid, normal)
        
        slice_ = slicedPart.section(plane_origin=centroid, plane_normal=normal)
        slice_2D, to_3D = slice_.to_planar()
        #slice_2D.show()

        if len(slice_2D.polygons_closed) > 0:
            # A sequence of connected vertices in space
            pgon = Polygon([[x, y] for x, y in slice_2D.discrete[0]])
            #area = np.round(pgon.area, 6)
            perimetro = pgon.length
            #print("Perimeter: ", perimetro*100)
            
            """
            area = np.round(slice_2D.polygons_closed[0].area, 6)
            perimetro = slice_2D.polygons_closed[0].length
            # print("Area: ", area)
            print("Perimeter: ", perimetro)
            """
        else:
            perimetro = 0

        if _.name == "muñeca" or _.name == "cadera" or _.name == "cintura":
            perimetros.append(perimetro)

    
    

    # measures = check_output(["./measure_model/measure", "aux.obj", file_measurements], shell=False)
    # print(measures.decode('utf-8'))
    tiempos.append(datetime.now() - start_time)
    click.secho("Duration: {} \n".format(datetime.now() - start_time), fg="yellow")
    tiempos.append(np.sum(tiempos))

    click.secho("Saving to CSV", fg="yellow")
    #save_data_csv(directory, perimetros, tiempos)
    for _ in range(0,len(tiempos)):
        print(tiempos[_])

if __name__ == "__main__":
    main()
