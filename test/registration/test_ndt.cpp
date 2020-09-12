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
    ndt.setTransformationEpsilon(1e-9);
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
    ndt.setTransformationEpsilon(1e-9);
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

#include <chrono>
#include <pcl/registration/ndt_double.h>
#include <pcl/registration/impl/ndt_double.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, NormalDistributionsTransform_Benchmark)
{
  return;
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

  std::cout << target->size() << " vs " << source->size() << std::endl;


  NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setTransformationEpsilon(1e-5);
  ndt.setMaximumIterations(50);
  ndt.setResolution(1.0);

  NormalDistributionsTransformDouble<pcl::PointXYZ, pcl::PointXYZ> ndt_double;
  ndt_double.setTransformationEpsilon(1e-5);
  ndt_double.setMaximumIterations(50);
  ndt_double.setResolution(1.0);

  pcl::PointCloud<pcl::PointXYZ> aligned;

  auto t1 = std::chrono::high_resolution_clock::now();  
  for(int i=0; i<100; i++) {
    ndt.setInputTarget(target);
    ndt.setInputSource(source);
    ndt.align(aligned);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "float:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;

  for(int i=0; i<100; i++) {
    ndt_double.setInputTarget(target);
    ndt_double.setInputSource(source);
    ndt_double.align(aligned);
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  std::cout << "double:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e6 << "[msec]" << std::endl;

  for(int i=0; i<100; i++) {
    ndt.setInputTarget(target);
    ndt.setInputSource(source);
    ndt.align(aligned);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  std::cout << "float:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e6 << "[msec]" << std::endl;

  for(int i=0; i<100; i++) {
    ndt_double.setInputTarget(target);
    ndt_double.setInputSource(source);
    ndt_double.align(aligned);
  }
  auto t5 = std::chrono::high_resolution_clock::now();
  std::cout << "double:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count() / 1e6 << "[msec]" << std::endl;

  std::cout << "---" << std::endl;
  std::cout << "float :" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;
  std::cout << "double:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e6 << "[msec]" << std::endl;
  std::cout << "float :" << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e6 << "[msec]" << std::endl;
  std::cout << "double:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count() / 1e6 << "[msec]" << std::endl;
}


int
main (int argc, char** argv)
{
  std::cout << "supported instructions:" << std::endl;
  std::cout << Eigen::SimdInstructionSetsInUse() << std::endl;

  testing::InitGoogleTest (&argc, argv);
  return (RUN_ALL_TESTS ());
}
