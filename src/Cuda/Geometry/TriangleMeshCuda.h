//
// Created by wei on 10/10/18.
//

#pragma once

#include "GeometryClasses.h"
#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Common/TransformCuda.h>

#include <Cuda/Container/ArrayCuda.h>

#include <Open3D/Geometry/TriangleMesh.h>
#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Geometry/KDTreeSearchParam.h"
#include <Open3D/Geometry/BoundingVolume.h>

#include <memory>

namespace open3d {
namespace cuda {
class TriangleMeshCudaDevice {
public:
    ArrayCudaDevice<Vector3f> vertices_;
    ArrayCudaDevice<Vector3f> vertex_normals_;
    ArrayCudaDevice<Vector3f> vertex_colors_;
    ArrayCudaDevice<Vector3i> triangles_;

public:
    VertexType type_;
    int max_vertices_;
    int max_triangles_;
};

class TriangleMeshCuda : public geometry::Geometry3D {
public:
    std::shared_ptr<TriangleMeshCudaDevice> device_ = nullptr;
    ArrayCuda<Vector3f> vertices_;
    ArrayCuda<Vector3f> vertex_normals_;
    ArrayCuda<Vector3f> vertex_colors_;
    ArrayCuda<Vector3i> triangles_;

public:
    VertexType type_;
    int max_vertices_;
    int max_triangles_;

public:
    TriangleMeshCuda();
    TriangleMeshCuda(VertexType type, int max_vertices, int max_triangles);
    TriangleMeshCuda(const TriangleMeshCuda &other);
    TriangleMeshCuda &operator=(const TriangleMeshCuda &other);
    ~TriangleMeshCuda() override;

    void Reset();
    void UpdateDevice();

    void Create(VertexType type, int max_vertices, int max_triangles);
    void Release();

    bool HasVertices() const;
    bool HasTriangles() const;
    bool HasVertexNormals() const;
    bool HasVertexColors() const;

    void Upload(geometry::TriangleMesh &mesh);
    std::shared_ptr<geometry::TriangleMesh> Download();

public:
    TriangleMeshCuda& Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    TriangleMeshCuda& Transform(const Eigen::Matrix4d &transformation) override;

    /* Temp */
    Eigen::Vector3d GetCenter() const override {return Eigen::Vector3d(0, 0, 0);};
    geometry::AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override {

        std::vector<Eigen::Vector3d> points;
        points.push_back(GetMinBound());
        points.push_back(GetMaxBound());
       return geometry::AxisAlignedBoundingBox::CreateFromPoints(points);}
//        return GetAxisAlignedBoundingBox(); };
    geometry::OrientedBoundingBox GetOrientedBoundingBox() const override {
        std::vector<Eigen::Vector3d> points;
        points.push_back(GetMinBound());
        points.push_back(GetMaxBound());
        return geometry::OrientedBoundingBox::CreateFromPoints(points);}
//        return GetOrientedBoundingBox(); };
    TriangleMeshCuda &Translate(const Eigen::Vector3d &translation,
                              bool relative = true) override {return *this;} ;
    TriangleMeshCuda &Scale(const double scale, bool center = true) override {return *this;};
    TriangleMeshCuda &Rotate(const Eigen::Vector3d &rotation,
                           bool center = true,
                           RotationType type = RotationType::XYZ) override { return *this; };
};

class TriangleMeshCudaKernelCaller {
public:
    static void GetMinBound(const TriangleMeshCuda &mesh,
                            ArrayCuda<Vector3f> &min_bound);

    static void GetMaxBound(const TriangleMeshCuda &mesh,
                            ArrayCuda<Vector3f> &max_bound);

    static void Transform(TriangleMeshCuda &mesh,
                          TransformCuda &transform);
};

__GLOBAL__
void GetMinBoundKernel(TriangleMeshCudaDevice mesh,
                       ArrayCudaDevice<Vector3f> min_bound);
__GLOBAL__
void GetMaxBoundKernel(TriangleMeshCudaDevice mesh,
                       ArrayCudaDevice<Vector3f> max_bound);
__GLOBAL__
void TransformKernel(TriangleMeshCudaDevice, TransformCuda transform);

} // cuda
} // open3d
