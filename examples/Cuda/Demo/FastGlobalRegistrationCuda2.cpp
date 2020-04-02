//
// Created by wei on 3/1/19.
//

#include <string>
#include <vector>
#include <iostream>

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>

#include "Utils.h"

using namespace open3d;
using namespace open3d::utility;
using namespace open3d::io;
using namespace open3d::visualization;

int FastGlobalRegistrationForPointClouds2(
    const std::string &source_ply_path,
    const std::string &target_ply_path) {

    SetVerbosityLevel(VerbosityLevel::Debug);

    clock_t time = clock();
    auto source = CreatePointCloudFromFile(source_ply_path);
    LogDebug("CreateSource {}\n", clock() - time); time = clock();
    auto target = CreatePointCloudFromFile(target_ply_path);
    LogDebug("CreateTarget {}\n", clock() - time); time = clock();

    auto source_down = source->VoxelDownSample(3.);
    LogDebug("DownsampleSource {}\n", clock() - time); time = clock();
    auto target_down = target->VoxelDownSample(3.);
    LogDebug("DownsampleTarget {}\n", clock() - time); time = clock();
std::cout<<"oasijdfoiasf" << std::endl;
    /** Load data **/
    static cuda::FastGlobalRegistrationCuda2 fgr;
    fgr.setSource(*source_down);
    LogDebug("setSource {}\n", clock() - time); time = clock();
    fgr.setTarget(*target_down);
    LogDebug("setTarget {}\n", clock() - time); time = clock();
    fgr.computeSourceFeatures(*source_down);
    LogDebug("compSF {}\n", clock() - time); time = clock();
    fgr.computeTargetFeatures(*target_down);
    LogDebug("compTF {}\n", clock() - time); time = clock();
    Eigen::Matrix<float,-1,-1,Eigen::RowMajor> sF = fgr.getSourceFeatures();
    LogDebug("getSF {}\n", clock() - time); time = clock();
    Eigen::Matrix<float,-1,-1,Eigen::RowMajor> tF = fgr.getTargetFeatures();
    LogDebug("getTF {}\n", clock() - time); time = clock();
    fgr.setSourceFeatures(sF);
    LogDebug("setSF {}\n", clock() - time); time = clock();
    fgr.setTargetFeatures(tF);
    LogDebug("setTF {}\n", clock() - time); time = clock();
    fgr.init();
    LogDebug("init {}\n", clock() - time); time = clock();

    std::cout<<"oasijdfoiasf" << std::endl;

    bool finished = false;
    int iter = 0, max_iter = 64;
    while (!finished) {
                fgr.DoSingleIteration(iter++);
                if (iter >= max_iter)
                    finished = true;
    }

    LogDebug("time {}\n", clock() - time);

    /** Prepare visualizer **/
    std::cout<<"fagaga" << std::endl;
    VisualizerWithCudaModule visualizer;
    std::cout<<"fagaga1" << std::endl;
    if (!visualizer.CreateVisualizerWindow("Fast Global Registration",
        640, 480,0, 0)) {
        LogWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    std::cout<<"fagaga2" << std::endl;
    visualizer.BuildUtilities();
    std::cout<<"fagaga3" << std::endl;
    visualizer.UpdateWindowTitle();
    std::cout<<"fagaga4" << std::endl;
    visualizer.AddGeometry(source_down);
    std::cout<<"fagaga4" << std::endl;
    visualizer.AddGeometry(target_down);
    std::cout<<"fagaga4" << std::endl;

    std::cout<<"oasijdfoiasf" << std::endl;

    auto source_cpu = *source_down;
    auto target_cpu = *target_down;
    /* Update geometry */
    *source_down = *fgr.source_.Download();
    if (source_cpu.HasColors()) {
        source_down->colors_ = source_cpu.colors_;
    }
    *target_down = *fgr.target_.Download();
    if (target_cpu.HasColors()) {
        target_down->colors_ = target_cpu.colors_;
    }
    visualizer.UpdateGeometry();
    visualizer.ResetViewPoint(true);

    std::cout<<"oasijdfoiasf" << std::endl;

//    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
//        if (finished) return false;

//        /* FGR (1 iteration) */
//        fgr.DoSingleIteration(iter++);

//        /* Update geometry */
//        *source_down = *fgr.source_.Download();
//        if (source_cpu.HasColors()) {
//            source_down->colors_ = source_cpu.colors_;
//        }
//        *target_down = *fgr.target_.Download();
//        if (target_cpu.HasColors()) {
//            target_down->colors_ = target_cpu.colors_;
//        }
//        vis->UpdateGeometry();

//        if (iter == 1) {
//            vis->ResetViewPoint(true);
//        }

//        /* Update flags */
//        if (iter >= max_iter)
//            finished = true;
//        return !finished;
//    });

    bool should_close = false;
    while (!should_close) {
        should_close = !visualizer.PollEvents();
    }
    visualizer.DestroyVisualizerWindow();
    std::cout<<"oasijdfoiasf" << std::endl;

    return 0;
}

int main(int argc, char **argv) {
    std::string source_path, target_path;
    if (argc > 2) {
        source_path = argv[1];
        target_path = argv[2];
    } else {
//        std::string test_data_path = "/media/wei/Data/data/redwood_simulated/livingroom1-clean/fragments_cuda";
//        source_path = test_data_path + "/fragment_005.ply";
//        target_path = test_data_path + "/fragment_008.ply";
        std::string test_data_path = "C:/Users/Seikowave/Desktop";
        source_path = test_data_path + "/test150 - Cloud2.ply";
        target_path = test_data_path + "/test151 - Cloud2.ply";
    }

    FastGlobalRegistrationForPointClouds2(source_path, target_path);

    return 0;
}
