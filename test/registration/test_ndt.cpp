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

using namespace pcl;
using namespace pcl::io;

PointCloud<PointXYZ>::Ptr cloud_source, cloud_target;

void test_ndt(NormalDistributionsTransform<PointXYZ, PointXYZ>& reg) {
  reg.setStepSize (0.05);
  reg.setResolution (0.025f);
  reg.setInputSource (cloud_source);
  reg.setInputTarget (cloud_target);
  reg.setMaximumIterations (50);
  reg.setTransformationEpsilon (1e-8);

  // Register
  PointCloud<PointXYZ> output;
  reg.align (output);
  EXPECT_EQ (output.size (), cloud_source->size ());
  EXPECT_LT (reg.getFitnessScore (), 0.001);
  Eigen::Matrix4f transform = reg.getFinalTransformation();

  // Check if the single thread result is consistent
  reg.setNumberOfThreads(4);
  reg.align(output);
  EXPECT_LT((reg.getFinalTransformation() - transform).array().abs().maxCoeff(), 1e-6);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransform_KDTREE)
{
  NormalDistributionsTransform<PointXYZ, PointXYZ> reg;
  reg.setNeighborSearchMethod(NeighborSearchMethod::KDTREE);
  test_ndt(reg);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransform_DIRECT1)
{
  NormalDistributionsTransform<PointXYZ, PointXYZ> reg;
  reg.setNeighborSearchMethod(NeighborSearchMethod::DIRECT1);
  test_ndt(reg);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransform_DIRECT7)
{
  NormalDistributionsTransform<PointXYZ, PointXYZ> reg;
  reg.setNeighborSearchMethod(NeighborSearchMethod::DIRECT7);
  test_ndt(reg);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransform_DIRECT27)
{
  NormalDistributionsTransform<PointXYZ, PointXYZ> reg;
  reg.setNeighborSearchMethod(NeighborSearchMethod::DIRECT27);
  test_ndt(reg);
}

#include <pcl/registration/ndt_old.h>
#include <pcl/registration/impl/ndt_old.hpp>
#include <pcl/filters/approximate_voxel_grid.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransform_Validation)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile("/home/koide/catkin_ws/src/fast_gicp/data/251370668.pcd", *target);
  pcl::io::loadPCDFile("/home/koide/catkin_ws/src/fast_gicp/data/251371071.pcd", *source);

  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  voxelgrid.setInputCloud(target);
  voxelgrid.filter(*filtered);
  *target = *filtered;

  voxelgrid.setInputCloud(source);
  voxelgrid.filter(*filtered);
  *source = *filtered;

  pcl::PointCloud<pcl::PointXYZ> aligned;

  {
    std::ofstream reg_ofs("/tmp/ndt.txt");
    std::ofstream reg_err_ofs("/tmp/ndt_cerr.txt");
    std::streambuf* cout_buf = std::cout.rdbuf();
    std::streambuf* cerr_buf = std::cerr.rdbuf();
    std::cout.rdbuf(reg_ofs.rdbuf());
    std::cerr.rdbuf(reg_err_ofs.rdbuf());

    NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setTransformationEpsilon(1e-5);
    ndt.setMaximumIterations(50);
    ndt.setNeighborSearchMethod(NeighborSearchMethod::KDTREE);
    ndt.setResolution(1.0);
    ndt.setInputTarget(target);
    ndt.setInputSource(source);
    ndt.align(aligned);

    std::cout << std::flush;
    std::cerr << std::flush;
    reg_ofs << std::flush;
    reg_err_ofs << std::flush;

    std::cout.rdbuf(cout_buf);
    std::cerr.rdbuf(cerr_buf);
  }

  {
    std::ofstream reg_ofs("/tmp/ndt_old.txt");
    std::ofstream reg_err_ofs("/tmp/ndt_old_cerr.txt");
    std::streambuf* cout_buf = std::cout.rdbuf();
    std::streambuf* cerr_buf = std::cerr.rdbuf();
    std::cout.rdbuf(reg_ofs.rdbuf());
    std::cerr.rdbuf(reg_err_ofs.rdbuf());

    NormalDistributionsTransformOld<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setTransformationEpsilon(1e-5);
    ndt.setMaximumIterations(50);
    ndt.setResolution(1.0);
    ndt.setInputTarget(target);
    ndt.setInputSource(source);
    ndt.align(aligned);

    std::cout << std::flush;
    std::cerr << std::flush;
    reg_ofs << std::flush;
    reg_err_ofs << std::flush;

    std::cout.rdbuf(cout_buf);
    std::cerr.rdbuf(cerr_buf);
  }
}


int
main (int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "No test files given. Please download `bun0.pcd` and `bun4.pcd`pass their path to the test." << std::endl;
    return (-1);
  }

  cloud_source.reset(new PointCloud<PointXYZ>);
  cloud_target.reset(new PointCloud<PointXYZ>);

  if (loadPCDFile (argv[1], *cloud_source) < 0)
  {
    std::cerr << "Failed to read test file. Please download `bun0.pcd` and pass its path to the test." << std::endl;
    return (-1);
  }
  if (loadPCDFile (argv[2], *cloud_target) < 0)
  {
    std::cerr << "Failed to read test file. Please download `bun4.pcd` and pass its path to the test." << std::endl;
    return (-1);
  }

  testing::InitGoogleTest (&argc, argv);
  return (RUN_ALL_TESTS ());
}
