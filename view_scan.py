import open3d as o3d

string = "individual_clouds/cloud_1706880075_285461000.pcd"
pcd = o3d.io.read_point_cloud("./" + string)
o3d.visualization.draw_geometries([pcd])


