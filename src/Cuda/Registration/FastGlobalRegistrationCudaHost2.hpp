//
// Created by wei on 1/21/19.
//

#include <Cuda/Geometry/NNCuda.h>
#include "FastGlobalRegistrationCuda2.h"
#include "Open3D/Utility/Console.h"
namespace open3d {
namespace cuda {
void FastGlobalRegistrationCuda2::Create() {
    utility::LogDebug("Creating2!\n");
    if (device_ == nullptr) {
        utility::LogDebug("Creating!\n");
        device_ = std::make_shared<FastGlobalRegistrationCudaDevice2>();
        results_.Create(28);
    }
}

void FastGlobalRegistrationCuda2::Release() {
    utility::LogDebug("Releasing2!\n");
    if (device_ != nullptr && device_.use_count() == 1) {
        utility::LogDebug("Releasing!\n");
        results_.Release();
        source_.Release();
        target_.Release();
        corres_source_to_target_.Release();
        corres_target_to_source_.Release();
        corres_mutual_.Release();
        corres_final_.Release();
        mSourceMaxSize = 0;
        mTargetMaxSize = 0;
        nn_source_to_target_.Release();
        nn_target_to_source_.Release();
    }
    device_ = nullptr;
}

void FastGlobalRegistrationCuda2::UpdateDevice() {
    if (device_ == nullptr) {
        utility::LogError("Server not initialized!\n");
        return;
    }

    device_->results_ = *results_.device_;
    device_->source_ = *source_.device_;
    device_->target_ = *target_.device_;
    device_->corres_source_to_target_ = *corres_source_to_target_.device_;
    device_->corres_target_to_source_ = *corres_target_to_source_.device_;
}

void FastGlobalRegistrationCuda2::ExtractResults(
    Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr, float &rmse) {
    std::vector<float> downloaded_result = results_.DownloadAll();
    int cnt = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            JtJ(i, j) = JtJ(j, i) = downloaded_result[cnt];
            ++cnt;
        }
    }
    for (int i = 0; i < 6; ++i) {
        Jtr(i) = downloaded_result[cnt];
        ++cnt;
    }
    rmse = downloaded_result[cnt];
}

void FastGlobalRegistrationCuda2::Initialize(
    geometry::PointCloud &source, geometry::PointCloud &target) {

    source_.Create(VertexWithNormal, (int)source.points_.size());
    source_.Upload(source);
    target_.Create(VertexWithNormal, (int)target.points_.size());
    target_.Upload(target);

    /* 0) Extract feature from original point clouds */
    source_feature_extractor_.Compute(
        source, geometry::KDTreeSearchParamHybrid(mFeatureRadius, mMaxFeatureNeighbors));
    target_feature_extractor_.Compute(
        target, geometry::KDTreeSearchParamHybrid(mFeatureRadius, mMaxFeatureNeighbors));
    source_features_ = source_feature_extractor_.fpfh_features_;
    target_features_ = target_feature_extractor_.fpfh_features_;

    /* 1) Initial Matching */
    nn_source_to_target_.BruteForceNN(source_features_, target_features_);
    corres_source_to_target_.SetCorrespondenceMatrix(
        nn_source_to_target_.nn_idx_);
    corres_source_to_target_.Compress();

    nn_target_to_source_.BruteForceNN(target_features_, source_features_);
    corres_target_to_source_.SetCorrespondenceMatrix(
        nn_target_to_source_.nn_idx_);
    corres_target_to_source_.Compress();
    UpdateDevice();

    /* 2) Reciprocity Test */
    corres_mutual_.Create(source.points_.size());
    device_->corres_mutual_ = *corres_mutual_.device_;
    FastGlobalRegistrationCudaKernelCaller2::ReciprocityTest(*this);

    /* 3) Tuple Test */
    corres_final_.Create(corres_mutual_.size() * 300);
    device_->corres_final_ = *corres_final_.device_;
    FastGlobalRegistrationCudaKernelCaller2::TupleTest(*this);

    double scale_global = NormalizePointClouds();
    device_->scale_global_ = (float) scale_global;
    device_->par_ = (float) scale_global;

    transform_normalized_source_to_target_ = Eigen::Matrix4d::Identity();
}

void FastGlobalRegistrationCuda2::InitializeSource(int size)
{
//    if (mSourceMaxSize < size) {
        mSourceMaxSize = size;
        source_.Create(VertexWithNormal, size);
//    } else {
//        utility::LogDebug("InitializeSource already inited > {}", size);
//    }
}

void FastGlobalRegistrationCuda2::InitializeTarget(int size)
{
//    if (mTargetMaxSize < size) {
        mTargetMaxSize = size;
        target_.Create(VertexWithNormal, size);
//    } else {
//        utility::LogDebug("InitializeTarget already inited > {}", size);
//    }
}

void FastGlobalRegistrationCuda2::setSource(geometry::PointCloud &source)
{
    if (source.points_.size() != mSourceMaxSize) {
//        source_.Release();
        InitializeSource(source.points_.size());
    }
//    source_.Reset();
    source_.Upload(source);
    mSourceSize = source.points_.size();
}

void FastGlobalRegistrationCuda2::setTarget(geometry::PointCloud &target)
{
    if (target.points_.size() != mTargetMaxSize) {
//        target_.Release();
        InitializeTarget(target.points_.size());
    }
//    target_.Reset();
    target_.Upload(target);
    mTargetSize = target.points_.size();
}

void FastGlobalRegistrationCuda2::computeSourceFeatures(geometry::PointCloud &source)
{
    source_feature_extractor_.Release();
    source_feature_extractor_.Create();
    source_feature_extractor_.Compute(
        source, geometry::KDTreeSearchParamHybrid(mFeatureRadius, mMaxFeatureNeighbors));
    source_features_ = source_feature_extractor_.fpfh_features_;
}

void FastGlobalRegistrationCuda2::computeTargetFeatures(geometry::PointCloud &target)
{
    target_feature_extractor_.Release();
    target_feature_extractor_.Create();
    target_feature_extractor_.Compute(
        target, geometry::KDTreeSearchParamHybrid(mFeatureRadius, mMaxFeatureNeighbors));
    target_features_ = target_feature_extractor_.fpfh_features_;
}

void FastGlobalRegistrationCuda2::setSourceFeatures(Eigen::Matrix<float, -1, -1, Eigen::RowMajor> sourceFeatures)
{
    source_features_.Release();
    source_features_.Create(sourceFeatures.rows(), sourceFeatures.cols());
    source_features_.Upload(sourceFeatures);
}

void FastGlobalRegistrationCuda2::setTargetFeatures(Eigen::Matrix<float, -1, -1, Eigen::RowMajor> targetFeatures)
{
    target_features_.Release();
    target_features_.Create(targetFeatures.rows(), targetFeatures.cols());
    target_features_.Upload(targetFeatures);
}

void FastGlobalRegistrationCuda2::init()
{
    /* 1) Initial Matching */
    if (nn_source_to_target_.device_ == nullptr) { nn_source_to_target_.Create(); }
    nn_source_to_target_.BruteForceNN(source_features_, target_features_);
    corres_source_to_target_.SetCorrespondenceMatrix(
        nn_source_to_target_.nn_idx_);
    corres_source_to_target_.Compress();

    if (nn_target_to_source_.device_ == nullptr) { nn_target_to_source_.Create(); }
    nn_target_to_source_.BruteForceNN(target_features_, source_features_);
    corres_target_to_source_.SetCorrespondenceMatrix(
        nn_target_to_source_.nn_idx_);
    corres_target_to_source_.Compress();
    UpdateDevice();

//    utility::LogError("2\n");
    /* 2) Reciprocity Test */
    corres_mutual_.Create(mSourceSize);
    device_->corres_mutual_ = *corres_mutual_.device_;
    FastGlobalRegistrationCudaKernelCaller2::ReciprocityTest(*this);

//    utility::LogError("3\n");
    /* 3) Tuple Test */
    corres_final_.Create(corres_mutual_.size() * 300);
    device_->corres_final_ = *corres_final_.device_;
    FastGlobalRegistrationCudaKernelCaller2::TupleTest(*this);

    double scale_global = NormalizePointClouds();
    device_->scale_global_ = (float) scale_global;
    device_->par_ = (float) scale_global;

    transform_normalized_source_to_target_ = Eigen::Matrix4d::Identity();
}


double FastGlobalRegistrationCuda2::NormalizePointClouds() {
    double scale_source, scale_target;

    std::tie(mean_source_, scale_source) = source_.Normalize();
    std::tie(mean_target_, scale_target) = target_.Normalize();
    double scale_global = std::max(scale_source, scale_target);
    utility::LogDebug("SCALES: {} {} // {}\n", scale_source, scale_target, scale_global);
    scale_global = 1.0;
    source_.Rescale(scale_global);
    target_.Rescale(scale_global);
    return scale_global;
}

namespace {
Eigen::Matrix4d GetTransformationOriginalScale2(
    const Eigen::Matrix4d &transformation,
    const Eigen::Vector3d &mean_source,
    const Eigen::Vector3d &mean_target,
    const double scale_global) {
    Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformation.block<3, 1>(0, 3);
    Eigen::Matrix4d transtemp = Eigen::Matrix4d::Zero();
    transtemp.block<3, 3>(0, 0) = R;
    transtemp.block<3, 1>(0, 3) =
        -R * mean_source + t * scale_global + mean_target;
    transtemp(3, 3) = 1;
    return transtemp;
}
} // unnamed namespace

RegistrationResultCuda FastGlobalRegistrationCuda2::DoSingleIteration(int iter) {
    RegistrationResultCuda result;
    result.transformation_ = Eigen::Matrix4d::Identity();
    result.inlier_rmse_ = 0;

    if (corres_final_.size() < 10) return result;

    results_.Memset(0);
    clock_t time = clock();
    FastGlobalRegistrationCudaKernelCaller2::
    ComputeResultsAndTransformation(*this);
//    std::cout << "FRCKC " << clock() - time << " " << iter << std::endl;; time = clock();
    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;
    float rmse;
    ExtractResults(JtJ, Jtr, rmse);
//    std::cout << "ExtractResults " << clock() - time << " " <<  iter << std::endl; time = clock();

    bool success;
    Eigen::VectorXd xi;
    std::tie(success, xi) = utility::SolveLinearSystemPSD(-JtJ, Jtr);
//    std::cout << "SolveLinearSystemPSD "<< clock() - time << " " <<  iter << std::endl; time = clock();
    Eigen::Matrix4d delta = utility::TransformVector6dToMatrix4d(xi);
//    std::cout << "TransformVector6dToMatrix4d "<< clock() - time << " " <<  iter << std::endl; time = clock();
    transform_normalized_source_to_target_ =
        delta * transform_normalized_source_to_target_;
    source_.Transform(delta);
//    std::cout << "tform "<< clock() - time << " " <<  iter << std::endl; time = clock();

    result.transformation_ = GetTransformationOriginalScale2(
        transform_normalized_source_to_target_,
        mean_source_, mean_target_,
        device_->scale_global_);
    result.inlier_rmse_ = rmse;
//    std::cout << "crateresu "<< clock() - time << " " <<  iter << std::endl; time = clock();
//    utility::LogDebug("Iteration {}: inlier rmse = {}\n", iter, rmse);

    if (iter % 4 == 0 && device_->par_ > 0.0f) {
        device_->par_ /= 1.4f;
    }

    return result;
};

RegistrationResultCuda FastGlobalRegistrationCuda2::ComputeRegistration() {
    RegistrationResultCuda result;
    for (int i = 0; i < 64; ++i) {
        result = DoSingleIteration(i);
    }
    return result;
}
}
}
