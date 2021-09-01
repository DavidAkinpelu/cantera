// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#ifndef CT_EIGEN_DEFS_H
#define CT_EIGEN_DEFS_H

#include "cantera/base/config.h"

// suppress warnings due to upstream issue in Eigen/src/Core/AssignEvaluator.h:
// warning: enum constant in boolean context [-Wint-in-bool-context]
// Eigen version 3.3.7
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif

#if CT_USE_SYSTEM_EIGEN
#include <Eigen/Dense>
#include <Eigen/Sparse>
#else
#include "cantera/ext/Eigen/Dense"
#include "cantera/ext/Eigen/Sparse"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace Cantera
{

typedef Eigen::Map<Eigen::MatrixXd> MappedMatrix;
typedef Eigen::Map<const Eigen::MatrixXd> ConstMappedMatrix;
typedef Eigen::Map<Eigen::VectorXd> MappedVector;
typedef Eigen::Map<const Eigen::VectorXd> ConstMappedVector;
typedef Eigen::Map<Eigen::RowVectorXd> MappedRowVector;
typedef Eigen::Map<const Eigen::RowVectorXd> ConstMappedRowVector;

typedef std::vector<Eigen::Triplet<double>> SparseTriplets;
}

#endif
