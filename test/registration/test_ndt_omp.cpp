/*
* Software License Agreement (BSD License)
*
*  Point Cloud Library (PCL) - www.pointclouds.org
*  Copyright (c) 2010-2011, Willow Garage, Inc.
*  Copyright (c) 2018-, Open Perception, Inc.
*
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the copyright holder(s) nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
* $Id$
*
*/

#include <pcl/test/gtest.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/ndt_omp.h>

using namespace pcl;
using namespace pcl::io;

PointCloud<PointXYZ> cloud_source, cloud_target;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransformOMP_DIRECT1)
{
  using PointT = PointXYZ;
  PointCloud<PointT>::Ptr src (new PointCloud<PointT>);
  copyPointCloud (cloud_source, *src);
  PointCloud<PointT>::Ptr tgt (new PointCloud<PointT>);
  copyPointCloud (cloud_target, *tgt);
  PointCloud<PointT> output;

  NormalDistributionsTransformOMP<PointT, PointT> reg;
  reg.setNeighborSearchMethod(NeighborSearchMethod::DIRECT1);
  reg.setStepSize (0.05);
  reg.setResolution (0.025f);
  reg.setInputSource (src);
  reg.setInputTarget (tgt);
  reg.setMaximumIterations (50);
  reg.setTransformationEpsilon (1e-8);
  // Register
  reg.align (output);
  EXPECT_EQ (int (output.size ()), int (cloud_source.size ()));
  EXPECT_LT (reg.getFitnessScore (), 0.001);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransformOMP_DIRECT7)
{
  using PointT = PointXYZ;
  PointCloud<PointT>::Ptr src (new PointCloud<PointT>);
  copyPointCloud (cloud_source, *src);
  PointCloud<PointT>::Ptr tgt (new PointCloud<PointT>);
  copyPointCloud (cloud_target, *tgt);
  PointCloud<PointT> output;

  NormalDistributionsTransformOMP<PointT, PointT> reg;
  reg.setNeighborSearchMethod(NeighborSearchMethod::DIRECT7);
  reg.setStepSize (0.05);
  reg.setResolution (0.025f);
  reg.setInputSource (src);
  reg.setInputTarget (tgt);
  reg.setMaximumIterations (50);
  reg.setTransformationEpsilon (1e-8);
  // Register
  reg.align (output);
  EXPECT_EQ (int (output.size ()), int (cloud_source.size ()));
  EXPECT_LT (reg.getFitnessScore (), 0.001);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransformOMP_DIRECT27)
{
  using PointT = PointXYZ;
  PointCloud<PointT>::Ptr src (new PointCloud<PointT>);
  copyPointCloud (cloud_source, *src);
  PointCloud<PointT>::Ptr tgt (new PointCloud<PointT>);
  copyPointCloud (cloud_target, *tgt);
  PointCloud<PointT> output;

  NormalDistributionsTransformOMP<PointT, PointT> reg;
  reg.setNeighborSearchMethod(NeighborSearchMethod::DIRECT26);
  reg.setStepSize (0.05);
  reg.setResolution (0.025f);
  reg.setInputSource (src);
  reg.setInputTarget (tgt);
  reg.setMaximumIterations (50);
  reg.setTransformationEpsilon (1e-8);
  // Register
  reg.align (output);
  EXPECT_EQ (int (output.size ()), int (cloud_source.size ()));
  EXPECT_LT (reg.getFitnessScore (), 0.001);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransformOMP_KDTREE)
{
  using PointT = PointXYZ;
  PointCloud<PointT>::Ptr src (new PointCloud<PointT>);
  copyPointCloud (cloud_source, *src);
  PointCloud<PointT>::Ptr tgt (new PointCloud<PointT>);
  copyPointCloud (cloud_target, *tgt);
  PointCloud<PointT> output;

  NormalDistributionsTransformOMP<PointT, PointT> ndt_omp;
  ndt_omp.setNeighborSearchMethod(NeighborSearchMethod::KDTREE);
  ndt_omp.setStepSize (0.05);
  ndt_omp.setResolution (0.025f);
  ndt_omp.setInputSource (src);
  ndt_omp.setInputTarget (tgt);
  ndt_omp.setMaximumIterations (50);
  ndt_omp.setTransformationEpsilon (1e-8);
  ndt_omp.align (output);

  EXPECT_EQ (int (output.size ()), int (cloud_source.size ()));
  EXPECT_LT (ndt_omp.getFitnessScore (), 0.001);

  NormalDistributionsTransform<PointT, PointT> ndt;
  ndt.setStepSize (0.05);
  ndt.setResolution (0.025f);
  ndt.setInputSource (src);
  ndt.setInputTarget (tgt);
  ndt.setMaximumIterations (50);
  ndt.setTransformationEpsilon (1e-8);
  ndt.align (output);

  EXPECT_LT ((ndt_omp.getFinalTransformation() - ndt.getFinalTransformation()).array().abs().maxCoeff(), 1e-6);
}

int
main (int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "No test files given. Please download `bun0.pcd` and `bun4.pcd`pass their path to the test." << std::endl;
    return (-1);
  }

  if (loadPCDFile (argv[1], cloud_source) < 0)
  {
    std::cerr << "Failed to read test file. Please download `bun0.pcd` and pass its path to the test." << std::endl;
    return (-1);
  }
  if (loadPCDFile (argv[2], cloud_target) < 0)
  {
    std::cerr << "Failed to read test file. Please download `bun4.pcd` and pass its path to the test." << std::endl;
    return (-1);
  }

  testing::InitGoogleTest (&argc, argv);
  return (RUN_ALL_TESTS ());
}
