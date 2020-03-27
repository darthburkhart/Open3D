// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>

#include "Utils.h"

using namespace open3d;
using namespace open3d::camera;
using namespace open3d::io;
using namespace open3d::geometry;
using namespace open3d::utility;

int main(int argc, char *argv[]) {
    using namespace open3d;
    SetVerbosityLevel(VerbosityLevel::Debug);
int jj = 0;
LogDebug("{}\n",jj++);
    std::string base_path = "D:/InfiniTAM Data/lounge";
    std::string base_path2 = "D:/InfiniTAM Data/lounge/frames2/";
    auto camera_trajectory = CreatePinholeCameraTrajectoryFromFile(base_path + "/lounge_trajectory.log");
    auto rgbd_filenames = ReadDataAssociation(base_path + "/data_association.txt");
    LogDebug("{}\n",jj++);
//    LogDebug("{} {}", rgbd_filenames[1].first, rgbd_filenames[1].second);
//    return 0;
    int index = 0;
    cuda::PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    LogDebug("{}\n",jj++);

    float voxel_length = .02f;
    cuda::TransformCuda extrinsics = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda tsdf_volume(
        8, voxel_length, 3 * voxel_length, extrinsics);
    LogDebug("{}\n",jj++);

    Image depth, color;
    cuda::RGBDImageCuda rgbd(640, 480, 5.f, 1000.0f);
    cuda::ScalableMeshVolumeCuda mesher(cuda::VertexWithNormalAndColor, 8,
                                        120000);
    LogDebug("{}\n",jj++);

    visualization::VisualizerWithCudaModule visualizer;
    if (!visualizer.CreateVisualizerWindow("ScalableFusion", 640, 480, 50, 50)) {
        LogWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    LogDebug("{}\n",jj++);
    visualizer.BuildUtilities();
    LogDebug("{}\n",jj++);
    visualizer.UpdateWindowTitle();
    LogDebug("{}\n",jj++);

    std::shared_ptr<geometry::TriangleMesh>
        mesh = std::make_shared<geometry::TriangleMesh>();
    ReadImage(base_path2 + rgbd_filenames[0].first, depth);
    ReadImage(base_path2 + rgbd_filenames[0].second, color);
    rgbd.Upload(depth, color);

    /* Use ground truth trajectory */
    Eigen::Matrix4d extrinsic =
            camera_trajectory->parameters_[0].extrinsic_.inverse();

    extrinsics.FromEigen(extrinsic);
    tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

    mesher.MarchingCubes(tsdf_volume);

    *mesh = *(mesher.mesh().Download());
//    *mesh = *(mesher.mesh().Download()->SamplePointsUniformly(10000));
    visualizer.AddGeometry(mesh);

    visualizer.UpdateGeometry();


    LogDebug("{}\n",jj++);

    Timer timer;
    for (int i = 1; i < rgbd_filenames.size() - 1; ++i) {
        LogDebug("Processing frame {} ...\n", index);
        ReadImage(base_path2 + rgbd_filenames[i].first, depth);
        ReadImage(base_path2 + rgbd_filenames[i].second, color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d extrinsic =
            camera_trajectory->parameters_[index].extrinsic_.inverse();

//        LogDebug("{} {} {} {}\n{} {} {} {}\n{} {} {} {}\n{} {} {} {}\n",extrinsic(0,0),extrinsic(1,0),extrinsic(2,0),extrinsic(3,0),
//                                                                        extrinsic(0,1),extrinsic(1,1),extrinsic(2,1),extrinsic(3,1),
//                                                                        extrinsic(0,2),extrinsic(1,2),extrinsic(2,2),extrinsic(3,2),
//                                                                        extrinsic(0,3),extrinsic(1,3),extrinsic(2,3),extrinsic(3,3)
//                                                                        );
        extrinsics.FromEigen(extrinsic);
        tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

        timer.Start();
        tsdf_volume.GetAllSubvolumes();
        mesher.MarchingCubes(tsdf_volume);
        timer.Stop();

        *mesh = *(mesher.mesh().Download());
        visualizer.PollEvents();
        visualizer.UpdateGeometry();
        visualizer.GetViewControl().ConvertFromPinholeCameraParameters(
            camera_trajectory->parameters_[index]);
        index++;
    }

    tsdf_volume.GetAllSubvolumes();
    mesher.MarchingCubes(tsdf_volume);
    *mesh = *mesher.mesh().Download();
    io::WriteTriangleMesh("copyroom.ply", *mesh);
//    io::WriteUniformTSDFVolumeToBIN("copyroom_uniform.bin", tsdf_volume);

}

