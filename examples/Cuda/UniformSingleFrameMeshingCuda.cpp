//
// Created by wei on 10/10/20.
//

#include <Cuda/Integration/UniformTSDFVolumeCuda.h>
#include <Cuda/Integration/UniformMeshVolumeCuda.h>
#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Common/VectorCuda.h>
#include <Core/Core.h>
#include <Eigen/Eigen>
#include <IO/IO.h>
#include <vector>

int main(int argc, char **argv) {
    using namespace open3d;
    Image depth, color;
    ReadImage("../../../examples/TestData/RGBD/depth/00000.png", depth);
    ReadImage("../../../examples/TestData/RGBD/color/00000.jpg", color);
    RGBDImageCuda rgbd(0.1f, 3.0f, 1000.0f);
    rgbd.Upload(depth, color);

    PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    TransformCuda transform = TransformCuda::Identity();

    const float voxel_length = 0.01f;
    transform.SetTranslation(Vector3f(-voxel_length * 256));
    UniformTSDFVolumeCuda<512> volume(voxel_length, voxel_length * 3, transform);

    TransformCuda extrinsics = TransformCuda::Identity();
    volume.Integrate(rgbd, intrinsics, extrinsics);
    UniformMeshVolumeCuda<512> mesher(VertexWithNormalAndColor, 400000, 800000);

    Timer timer;
    timer.Start();
    int iters = 100;
    for (int i = 0; i < iters; ++i) {
        mesher.MarchingCubes(volume);
    }
    timer.Stop();
    PrintInfo("MarchingCubes time: %f milliseconds\n",
        timer.GetDuration() / iters);

    WriteTriangleMeshToPLY("test_uniform.ply", *mesher.mesh().Download());
}