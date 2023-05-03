import pathlib
from typing import List, Tuple
from copy import deepcopy
import yaml

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# TODO add output folder, add report, read file in main

color_map = plt.get_cmap('tab20')


def read_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd_path = pathlib.Path(path)
    if not pcd_path.exists():
        return print('pcd file not found')
    return o3d.io.read_point_cloud(str(pcd_path))


def read_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh_path = pathlib.Path(path)
    if not mesh_path.exists():
        return print('mesh file not found')
    return o3d.io.read_triangle_mesh(str(mesh_path))


def statistical_outlier_removal(pcd: o3d.geometry.PointCloud,
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0,
                                ) -> List[o3d.geometry.PointCloud]:

    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])

    return [inlier_cloud, outlier_cloud]


def radius_outlier_removal(pcd: o3d.geometry.PointCloud,
                           nb_points: int = 16,
                           radius=0.05) -> List[o3d.geometry.PointCloud]:

    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=radius)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])

    return [inlier_cloud, outlier_cloud]


def segment_plane(pcd: o3d.geometry.PointCloud,
                  distance_threshold: float = 0.01
                  ):

    plane_equitation, ind = pcd.segment_plane(distance_threshold=distance_threshold,
                                              ransac_n=3,
                                              num_iterations=1000)

    [a, b, c, d] = plane_equitation
    report = {'a': str(a), 'b': str(b), 'c': str(c), 'd': str(d)}

    on_plane = pcd.select_by_index(ind)

    return [on_plane, pcd.select_by_index(ind, invert=True), report]


def segment_planes(pcd: o3d.geometry.PointCloud,
                   iterations: int = 3,
                   distance_threshold: float = 0.05):
    planes = []
    planes_viz = []
    report = {}
    rest = pcd

    for i in range(iterations):

        rest, plane, pq = segment_plane(rest, distance_threshold)
        planes.append(plane)
        plane_viz = deepcopy(plane)
        colors = color_map(i / iterations)
        plane_viz.paint_uniform_color(list(colors[:3]))
        planes_viz.append(plane_viz)
        report['plane_' + str(i)] = pq

    return planes, planes_viz, rest, report


def segment_clusters(pcd: o3d.geometry.PointCloud,
                     eps: float = 0.02,
                     min_points: int = 100,
                     ) -> List[o3d.geometry.PointCloud]:

    clusters = []
    clusters_vis = []

    cl = np.array(pcd.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=True))
    max_clusters = np.max(cl)

    # print(f'point cloud has {max_clusters + 1} clusters')
    # print(f'cluster labels: {np.unique(cl)}')
    # print(f'cluster sizes: {np.bincount(cl + 1)}')

    for i in range(max_clusters + 1):
        cluster = pcd.select_by_index(np.where(cl == i)[0])
        clusters.append(cluster)
        cluster_viz = deepcopy(cluster)
        cluster_viz.paint_uniform_color(color_map(i / (max_clusters + 1))[:3])
        clusters_vis.append(cluster_viz)

    clusters.sort(key=lambda x: len(x.points), reverse=True)
    clusters_vis.sort(key=lambda x: len(x.points), reverse=True)

    report = {'number_of_clusters': str(max_clusters),
              'cluster_sizes': str(np.bincount(cl + 1)).strip('[]')}

    return clusters, clusters_vis, report


def flatten_list(list_2d: List[List]) -> List:
    return [item for sublist in list_2d for item in sublist]


def write_report_yaml(report: dict, path: str) -> bool:
    with open(path, 'w') as file:
        documents = yaml.dump(report, file)
    return True


def segment_blocks(path: str,
                   num_blocks: int = 4,
                   write_meshes: bool = False,
                   generate_report: bool = True,
                   ground_plane_threshold: float = 0.003,
                   cluster_eps: float = 0.01,
                   cluster_min_points: int = 200,
                   ) -> bool:

    report = {}

    # read pcd file
    # ----------------
    pcd = read_pcd(path)
    if not pcd:
        return
    print('pcd read successfully')

    o3d.visualization.draw_geometries([pcd])

    # segment ground plane
    # ----------------
    plane, pcd, pq = segment_plane(pcd, ground_plane_threshold)
    if generate_report:
        report['ground_plane'] = pq
    print('ground plane segmented successfully')

    plane.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd, plane])

    # cluster
    # ----------------
    clusters, clusters_viz, r = segment_clusters(
        pcd, eps=cluster_eps, min_points=cluster_min_points)
    print(
        f'clusters segmented successfully, found {r["number_of_clusters"]} clusters with lengths {r["cluster_sizes"]}')
    clusters = clusters[:num_blocks]
    if generate_report:
        report['clusters'] = r

    o3d.visualization.draw_geometries(clusters_viz)
    o3d.visualization.draw_geometries(clusters)

    # outlier removal
    # ----------------
    outliers = []
    for cluster in clusters:
        cluster, outlier_s = statistical_outlier_removal(cluster)
        outliers.append(outlier_s)
    outlier_removal = clusters + outliers
    print('outlier removal completed successfully')

    o3d.visualization.draw_geometries(outlier_removal)

    # cluster bounding box
    # ----------------
    # bb = []
    # for cluster in clusters:
    #     bb.append(o3d.geometry.OrientedBoundingBox.create_from_points(cluster.points))
    # cluster_bb = clusters + bb

    # o3d.visualization.draw_geometries(cluster_bb)

    # segment planes in clusters
    # ----------------
    cl_planes = []
    cl_planes_viz = []
    for idx, cluster in enumerate(clusters):
        planes, planplanees_viz, rest, r = segment_planes(cluster, 4, 0.002)
        cl_planes.append(planes)
        cl_planes_viz.append(planes_viz)
        if generate_report:
            report['clusters'].update({f'cluster_{idx}': r})
    print('iterative plane segmentation completed successfully')

    o3d.visualization.draw_geometries(flatten_list(cl_planes_viz))
    o3d.visualization.draw_geometries(flatten_list(cl_planes))

    # down sample cluster planes
    # ----------------
    # for idx, cl in enumerate(cl_planes):
    #     cl_planes[idx] = [plane.voxel_down_sample(0.005) for plane in cl]
    # flat_list = [
    #     item for sublist in cl_planes for item in sublist]
    # o3d.visualization.draw_geometries(flat_list)

    # extract center largest plane
    # ----------------
    centers = []
    bbs = []
    for idx, cl in enumerate(cl_planes):
        center = cl[0].get_center()
        centers.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=center))
        bbs.append(
            o3d.geometry.OrientedBoundingBox.create_from_points(cl[0].points))
        if generate_report:
            report['clusters'][f'cluster_{idx}']['plane_0'].update(
                {'center': {'x': str(center[0]), 'y': str(center[1]), 'z': str(center[2])}})

    if generate_report:
        write_report_yaml(report, '/home/v/report.yaml')

    o3d.visualization.draw_geometries(centers + bbs + flatten_list(cl_planes))

    # build mesh
    # ----------------
    cluster_planes_meshes = []
    for cl in cl_planes:
        mesh_planes = []
        for plane in cl:
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            #     plane, alpha=0.001)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(plane,
                                                                                   o3d.utility.DoubleVector(
                                                                                       [0.001, 0.005]))
            mesh = mesh.compute_vertex_normals()
            mesh_planes.append(mesh)
        cluster_planes_meshes.append(mesh_planes)

    o3d.visualization.draw_geometries(flatten_list(cluster_planes_meshes))

    # smooth mesh
    # ----------------
    for idx, cl in enumerate(cluster_planes_meshes):
        for jdx, mesh in enumerate(cl):
            mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
            mesh = mesh.compute_vertex_normals()
            mesh = mesh.normalize_normals()
            cluster_planes_meshes[idx][jdx] = mesh

    o3d.visualization.draw_geometries(flatten_list(cluster_planes_meshes))

    # write meshes to file
    # ----------------
    if write_meshes:
        mesh_path = pathlib.Path('/home/v/meshes')
        if not mesh_path.exists():
            mesh_path.mkdir()
        for idx, cl in enumerate(cluster_planes_meshes):
            for jdx, mesh in enumerate(cl):
                o3d.io.write_triangle_mesh(
                    f'{str(mesh_path)}/cluster{idx}_face{jdx}.ply', mesh)
        print('mesh written successfully')


def segment_shards(path,
                   num_blocks=9,
                   write_meshes=False,
                   generate_report=False,
                   ground_plane_threshold=0.008,
                   cluster_eps=0.02,
                   cluster_min_points=200,
                   cluster_plane_threshold=0.002):

    report = {}
    # read point cloud
    # ----------------
    pcd = read_pcd(path)
    if not pcd:
        return
    print('pcd read successfully')

    o3d.visualization.draw_geometries([pcd])

    # segment ground plane
    # ----------------
    plane, pcd, pq = segment_plane(pcd, ground_plane_threshold)
    if generate_report:
        report['ground_plane'] = pq
    print('ground plane segmented successfully')

    plane.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pcd, plane])

    # cluster
    # ----------------
    clusters, clusters_viz, r = segment_clusters(
        pcd, eps=cluster_eps, min_points=200)
    print(
        f'clusters segmented successfully, found {r["number_of_clusters"]} clusters with lengths {r["cluster_sizes"]}')
    clusters = clusters[:num_blocks]
    if generate_report:
        report['clusters'] = r

    # o3d.visualization.draw_geometries(clusters_viz)
    # o3d.visualization.draw_geometries(clusters)

    # plane segmentation
    # ----------------
    cl_planes = []
    outliers = []
    for cluster in clusters:
        cl_plane, outlier, r = segment_plane(cluster, cluster_plane_threshold)
        cl_planes.append(cl_plane)
        outliers.append(outlier.paint_uniform_color([1, 0, 0]))

    # o3d.visualization.draw_geometries(cl_planes + outliers)

    # planes center
    centers = []
    for plane in cl_planes:
        center = plane.get_center()
        centers.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=center))

    o3d.visualization.draw_geometries(centers + clusters)

    # outlier removal
    outliers = []
    for cluster in clusters:
        cluster, outlier = statistical_outlier_removal(cluster)
        outliers.append(outlier.paint_uniform_color([1, 0, 0]))

    o3d.visualization.draw_geometries(clusters + outliers)

    # build mesh
    # ----------------
    meshes = []
    for cluster in clusters:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cluster,
                                                                               o3d.utility.DoubleVector(
                                                                                   [0.001, 0.005]))
        mesh = mesh.compute_vertex_normals()
        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        mesh = mesh.compute_vertex_normals()
        mesh = mesh.normalize_normals()
        meshes.append(mesh)

    o3d.visualization.draw_geometries(meshes)

    # write meshes to file
    # ----------------
    if write_meshes:
        mesh_path = pathlib.Path('/home/v/meshes')
        if not mesh_path.exists():
            mesh_path.mkdir()
        for idx, cluster in enumerate(clusters):
            o3d.io.write_triangle_mesh(
                f'{str(mesh_path)}/cluster{idx}.ply', mesh)
        print('mesh written successfully')


if __name__ == '__main__':
    # segment_blocks('/home/v/test.ply', write_meshes=False, generate_report=True)
    segment_shards('/home/v/granite_shards_highres.ply',
                   write_meshes=False, generate_report=True)
