from pyntcloud import PyntCloud

human_face = PyntCloud.from_file("pointcloud.ply")

human_face.plot()
