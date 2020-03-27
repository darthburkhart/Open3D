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
#include <Cuda/IO/ClassIO/UniformTSDFVolumeCudaIO.h>

#include "Utils.h"

using namespace open3d;
using namespace open3d::utility;
using namespace open3d::io;
using namespace open3d::camera;
using namespace open3d::geometry;
using namespace open3d::visualization;

std::shared_ptr<Image> ConvertImageFromFloatImage(const Image &image) {
    auto uimage = std::make_shared<geometry::Image>();
    if (image.IsEmpty()) {
        return uimage;
    }

    uimage->Prepare(image.width_, image.height_, image.num_of_channels_,
                         1);

    int num_pixels = image.height_ * image.width_;
    for (int i = 0; i < num_pixels; i++) {
        uint8_t *p = (uint8_t *)(uimage->data_.data() +
                                 i * uimage->num_of_channels_);

        float *pf = (float *)(image.data_.data() +
                              i * image.num_of_channels_ *
                                      image.bytes_per_channel_);

        for (int k = 0; k < image.num_of_channels_; ++k) {
            p[k] = uint8_t(std::abs(pf[k]));
        }
    }

    return uimage;
}

int main(int argc, char *argv[]) {
    SetVerbosityLevel(VerbosityLevel::Debug);

    SetVerbosityLevel(VerbosityLevel::Debug);
int jj = 0;
LogDebug("{}\n",jj++);
    std::string base_path = "D:/InfiniTAM Data/lounge";
    std::string base_path2 = "D:/InfiniTAM Data/lounge/frames2/";
    auto camera_trajectory = CreatePinholeCameraTrajectoryFromFile(
            base_path + "/lounge_trajectory.log");
    auto rgbd_filenames =
            ReadDataAssociation(base_path + "/data_association.txt");

    FPSTimer timer("Process RGBD stream",
                   (int)camera_trajectory->parameters_.size());

    cuda::PinholeCameraIntrinsicCuda intrinsics(
            PinholeCameraIntrinsicParameters::PrimeSenseDefault);

//    float voxel_length = 0.008f;
    float voxel_length = 0.1f;
    int voxel_resolution = 256;
    float offset = -voxel_length * voxel_resolution / 2;
    cuda::TransformCuda extrinsics = cuda::TransformCuda::Identity();
    extrinsics.SetTranslation(cuda::Vector3f(offset, offset, offset));
    std::cout << extrinsics.ToEigen() << "\n";
    cuda::UniformTSDFVolumeCuda tsdf_volume(voxel_resolution, voxel_length,
                                            3 * voxel_length, extrinsics);
    cuda::UniformMeshVolumeCuda mesher(cuda::VertexWithNormalAndColor,
                                       voxel_resolution, 4000000, 8000000);

    Image depth, color;
    cuda::RGBDImageCuda rgbd(640, 480, 2.0f, 1000.0f);

    VisualizerWithCudaModule visualizer;
    if (!visualizer.CreateVisualizerWindow("UniformFusion", 640, 480, 0, 0)) {
        LogWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    std::shared_ptr<PointCloud> mesh = std::make_shared<PointCloud>();
    std::shared_ptr<TriangleMesh> mesh2 = std::make_shared<TriangleMesh>();


    ReadImage(base_path2 + rgbd_filenames[0].first, depth);
    ReadImage(base_path2 + rgbd_filenames[0].second, color);
    rgbd.Upload(depth, color);

    /* Use ground truth trajectory */
    Eigen::Matrix4d extrinsic =
            camera_trajectory->parameters_[0].extrinsic_.inverse();

    extrinsics.FromEigen(extrinsic);
    tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

    mesher.MarchingCubes(tsdf_volume);

    *mesh2 = *(mesher.mesh().Download());
//    *mesh = *(mesher.mesh().Download()->SamplePointsUniformly(10000));
    visualizer.AddGeometry(mesh2);

    visualizer.UpdateGeometry();



//    bool finished = false;
//    int iter = 1, max_iter = 1200;
//    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
//        if (finished) return false;

//        ReadImage(base_path2 + rgbd_filenames[iter].first, depth);
//        ReadImage(base_path2 + rgbd_filenames[iter].second, color);
//        rgbd.Upload(depth, color);

//        /* Use ground truth trajectory */
//        Eigen::Matrix4d extrinsic =
//                camera_trajectory->parameters_[iter].extrinsic_.inverse();

//        extrinsics.FromEigen(extrinsic);
//        tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

//        mesher.MarchingCubes(tsdf_volume);

//        *mesh2 = *(mesher.mesh().Download());

//        vis->UpdateGeometry();

//        iter++;
//        /* Update flags */
//        if (iter >= max_iter)
//            finished = true;
//        return !finished;
//    });

    for (int i = 1; i < 3003; ++i) {
        LogDebug("Processing frame {} ...\n", i);
        std::cout << base_path2 + rgbd_filenames[i].first << "\n";
        std::cout << base_path2 + rgbd_filenames[i].second << "\n";
        ReadImage(base_path2 + rgbd_filenames[i].first, depth);
        ReadImage(base_path2 + rgbd_filenames[i].second, color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d extrinsic =
                camera_trajectory->parameters_[i].extrinsic_.inverse();

        extrinsics.FromEigen(extrinsic);
        tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

        mesher.MarchingCubes(tsdf_volume);

        *mesh2 = *mesher.mesh().Download();
        std::cout << mesh2->vertices_.size() << "\n";

        visualizer.PollEvents();

        visualizer.UpdateGeometry();
        visualizer.GetViewControl().ConvertFromPinholeCameraParameters(
                camera_trajectory->parameters_[i]);
    }

    mesher.MarchingCubes(tsdf_volume);
    *mesh2 = *mesher.mesh().Download();
    io::WriteTriangleMesh("copyroom.ply", *mesh2);
    io::WriteUniformTSDFVolumeToBIN("copyroom_uniform.bin", tsdf_volume);


    bool should_close = false;
    while (!should_close) {
        should_close = !visualizer.PollEvents();
    }
//    io::ReadUniformTSDFVolumeFromBIN("copyroom_uniform_saved.bin", tsdf_volume);
//    mesher.MarchingCubes(tsdf_volume);
//
//    *mesh = mesher.mesh();
//    visualization::DrawGeometriesWithCudaModule({mesh});
//
//    io::WriteTriangleMesh("res.ply", *mesh->Download());

//    cuda::ImageCuda<float, 3> im_ray_casting(640, 480);
//    Eigen::Matrix4d extrinsic =
//            camera_trajectory->parameters_[0].extrinsic_.inverse();
//    extrinsics.FromEigen(extrinsic);
//
//    tsdf_volume.RayCasting(im_ray_casting, intrinsics, extrinsics);
//    io::WriteImage("test.png", *ConvertImageFromFloatImage(
//                                       *im_ray_casting.DownloadImage()));

    return 0;
}
