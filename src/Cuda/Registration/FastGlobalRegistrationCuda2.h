//
// Created by wei on 1/21/19.
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>
#include <Open3D/Geometry/PointCloud.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/FeatureExtractorCuda.h>
#include <Cuda/Registration/RegistrationCuda.h>
#include <Cuda/Geometry/NNCuda.h>
#include <memory>
#include <iostream>

namespace open3d {
namespace cuda {

class FastGlobalRegistrationCuda2;

class FastGlobalRegistrationCudaKernelCaller2 {
public:
    static void Create(cudaStream_t &stream);
    static void Destroy(cudaStream_t &stream);
    static cudaStream_t GetStream();
    static void ReciprocityTest(FastGlobalRegistrationCuda2 &fgr);
    static void TupleTest(FastGlobalRegistrationCuda2 &fgr);
    static void ComputeResultsAndTransformation(
        FastGlobalRegistrationCuda2 &fgr);
};

class FastGlobalRegistrationCudaDevice2 {

public:
    FastGlobalRegistrationCudaDevice2() {
    }
    ~FastGlobalRegistrationCudaDevice2() {
    }
    cudaStream_t regStream;


    PointCloudCudaDevice source_;
    PointCloudCudaDevice target_;

    Array2DCudaDevice<float> source_features_;
    Array2DCudaDevice<float> target_features_;

    CorrespondenceSetCudaDevice corres_source_to_target_;
    CorrespondenceSetCudaDevice corres_target_to_source_;

    ArrayCudaDevice<Vector2i> corres_mutual_;
    ArrayCudaDevice<Vector2i> corres_final_;

    ArrayCudaDevice<float> results_;

    float par_;
    float scale_global_;

    __DEVICE__
    void ComputePointwiseJacobianAndResidual(
        int source_idx, int target_idx,
        Vector6f &jacobian_x, Vector6f &jacobian_y, Vector6f &jacobian_z,
        Vector3f &residual, float &lij);
};

class FastGlobalRegistrationCuda2 {
public:
    std::shared_ptr<FastGlobalRegistrationCudaDevice2> device_ = nullptr;

public:
    FastGlobalRegistrationCuda2() { Create(); }
    ~FastGlobalRegistrationCuda2() { Release(); }
    void Create();
    void Release();

    void UpdateDevice();
    void ExtractResults(
        Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr, float &rmse);

public:

    void Initialize(geometry::PointCloud& source, geometry::PointCloud &target);
    double NormalizePointClouds();
    void AdvancedMatching();
    RegistrationResultCuda DoSingleIteration(int iter);
    RegistrationResultCuda ComputeRegistration();

public:
    PointCloudCuda source_;
    PointCloudCuda target_;

    FeatureExtractorCuda source_feature_extractor_;
    FeatureExtractorCuda target_feature_extractor_;

    Array2DCuda<float> source_features_;
    Array2DCuda<float> target_features_;

    NNCuda nn_source_to_target_;
    NNCuda nn_target_to_source_;

    CorrespondenceSetCuda corres_source_to_target_;
    CorrespondenceSetCuda corres_target_to_source_;

    ArrayCuda<Vector2i> corres_mutual_;
    ArrayCuda<Vector2i> corres_final_;

    ArrayCuda<float> results_;

    Eigen::Vector3d mean_source_;
    Eigen::Vector3d mean_target_;

    Eigen::Matrix4d transform_normalized_source_to_target_;

public:
    void InitializeSource(int size);
    void InitializeTarget(int size);
    void setSource(geometry::PointCloud &source);
    void setTarget(geometry::PointCloud &target);
    void computeSourceFeatures(geometry::PointCloud &source);
    void computeTargetFeatures(geometry::PointCloud &target);
    void setSourceFeatures(Eigen::Matrix<float, -1, -1, Eigen::RowMajor> sourceFeatures);
    void setTargetFeatures(Eigen::Matrix<float, -1, -1, Eigen::RowMajor> targetFeatures);
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> getSourceFeatures() { return source_features_.Download();}
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> getTargetFeatures() { return target_features_.Download();}

    void init();


private:
    int mTargetMaxSize = 0;
    int mSourceMaxSize = 0;
    int mTargetSize = 0;
    int mSourceSize = 0;

    int mMaxFeatureNeighbors = 1000;
    float mFeatureRadius = 10.0f;
};


__GLOBAL__
void ReciprocityTestKernel2(FastGlobalRegistrationCudaDevice2 server);
__GLOBAL__
void TupleTestKernel2(FastGlobalRegistrationCudaDevice2 server,
                     ArrayCudaDevice<float> random_numbers,
                     int tuple_tests);
__GLOBAL__
void ComputeResultsAndTransformationKernel2(FastGlobalRegistrationCudaDevice2 server);

} // cuda
} // open3d


