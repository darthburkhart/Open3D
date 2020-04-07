//
// Created by wei on 1/21/19.
//

#include "NNCuda.h"

namespace open3d {
namespace cuda {

NNCuda::NNCuda() {
    Create();
}

NNCuda::~NNCuda() {
    Release();
}

void NNCuda::Create() {
    if (device_ == nullptr) {
        device_ = std::make_shared<NNCudaDevice>();
    }
}
void NNCuda::Release() {
    if (device_ != nullptr) {
        query_.Release();
        reference_.Release();

        nn_idx_.Release();
        nn_dist_.Release();

        distance_matrix_.Release();
    }
    device_ = nullptr;
}
void NNCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->query_ = *query_.device_;
        device_->ref_ = *reference_.device_;
        device_->nn_idx_ = *nn_idx_.device_;
        device_->nn_dist_ = *nn_dist_.device_;
        device_->distance_matrix_ = *distance_matrix_.device_;
    }
}

void NNCuda::BruteForceNN(Eigen::MatrixXd &query, Eigen::MatrixXd &reference) {

    /** Change storage format for Array2DCuda **/
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> query_rowmajor;
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> reference_rowmajor;
    query_rowmajor = query.cast<float>();
    reference_rowmajor = reference.cast<float>();

    query_.Upload(query_rowmajor);
    reference_.Upload(reference_rowmajor);

    nn_idx_.Create(1, query_.max_cols_);
    nn_dist_.Create(1, query_.max_cols_);
    distance_matrix_.Create(reference_.max_cols_, query_.max_cols_);

    UpdateDevice();

    NNCudaKernelCaller::ComputeDistances(*this);
    NNCudaKernelCaller::FindNN(*this);
}

void NNCuda::BruteForceNN(Array2DCuda<float> &query,
                          Array2DCuda<float> &reference) {
    query_ = query;
    reference_ = reference;

    nn_idx_.Create(1, query_.max_cols_);
    nn_dist_.Create(1, query_.max_cols_);
    distance_matrix_.Create(reference_.max_cols_, query_.max_cols_);

    UpdateDevice();

    NNCudaKernelCaller::ComputeDistances(*this);
    NNCudaKernelCaller::FindNN(*this);
}
} // cuda
} // open3d
