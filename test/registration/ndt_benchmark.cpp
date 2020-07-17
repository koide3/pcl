#include <chrono>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/ndt_omp.h>
#include <pcl/filters/approximate_voxel_grid.h>


pcl::PointCloud<pcl::PointXYZ>::ConstPtr read_kitti_pointcloud(const std::string& filename) {
  FILE* file = fopen(filename.c_str(), "rb");
  if(!file) {
    std::cerr << "error: failed to load " << filename << std::endl;
    return nullptr;
  }

  std::vector<float> buffer(1000000);
  size_t num_points = fread(reinterpret_cast<char*>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
  fclose(file);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  cloud->resize(num_points);

  for(int i = 0; i < num_points; i++) {
    auto& pt = cloud->at(i);
    pt.x = buffer[i * 4];
    pt.y = buffer[i * 4 + 1];
    pt.z = buffer[i * 4 + 2];
    // pt.intensity = buffer[i * 4 + 3];
  }

  return cloud;
}

void run_kitti(const std::vector<pcl::PointCloud<pcl::PointXYZ>::ConstPtr>& clouds, pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>& reg, const std::string& method_name) {
  std::vector<double> times;
  times.reserve(clouds.size());

  std::cout << clouds.size() << std::endl;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses(clouds.size());
  poses[0].setIdentity();

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  for(int i=1; i < clouds.size(); i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    reg.setInputTarget(clouds[i - 1]);
    reg.setInputSource(clouds[i]);
    reg.align(*aligned);
    auto t2 = std::chrono::high_resolution_clock::now();

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9);
    poses[i] = poses[i - 1] * reg.getFinalTransformation();

    std::cout << i << "/" << clouds.size() << " : " << times.back() * 1e3 << "[msec]" << std::endl;
  }


  std::ofstream times_ofs("/home/koide/ndt_benchmark/" + method_name + "_times.csv");
  for(auto t: times) {
    times_ofs << t << std::endl;
  }

  std::ofstream poses_ofs("/home/koide/ndt_benchmark/" + method_name + "_poses.csv");
  for(const auto& pose: poses) {
    for(int i=0; i<3; i++) {
      for(int j=0; j<4; j++) {
        if(i || j) {
          poses_ofs << " ";
        }

        poses_ofs << pose(i, j);
      }
    }
    poses_ofs << std::endl;
  }

}

int
main (int argc, char** argv)
{
  std::cout << "loading KITTI" << std::endl;
  std::string kitti_path = "/home/koide/datasets/kitti/sequences/00/velodyne";
  boost::filesystem::directory_iterator itr(kitti_path);
  boost::filesystem::directory_iterator end;

  std::vector<std::string> filenames;
  for(itr; itr != end; itr++) {
    if(itr->path().extension() == ".bin") {
      filenames.push_back(itr->path().string());
    }
  }

  std::sort(filenames.begin(), filenames.end());
  // filenames.erase(filenames.begin() + 1024, filenames.end());

  std::vector<pcl::PointCloud<pcl::PointXYZ>::ConstPtr> clouds;
  for(const auto& filename: filenames) {
    clouds.push_back(read_kitti_pointcloud(filename));
  }

  std::cout << "downsampling..." << std::endl;
  #pragma omp parallel for
  for(int i=0; i<clouds.size(); i++) {
    double resolution = 0.25;
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(resolution, resolution, resolution);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid.setInputCloud(clouds[i]);
    voxelgrid.filter(*filtered);

    clouds[i] = filtered;
  }

  double ndt_resolution = 2.0;
  double translation_epsilon = 1e-3;

  pcl::NormalDistributionsTransformOMP<pcl::PointXYZ, pcl::PointXYZ> ndt_omp;
  ndt_omp.setResolution(ndt_resolution);
  ndt_omp.setTransformationEpsilon(translation_epsilon);

  ndt_omp.setNeighborSearchMethod(pcl::NeighborSearchMethod::DIRECT27);
  run_kitti(clouds, ndt_omp, "ndt_omp_direct27");

  ndt_omp.setNeighborSearchMethod(pcl::NeighborSearchMethod::DIRECT26);
  run_kitti(clouds, ndt_omp, "ndt_omp_direct26");

  ndt_omp.setNeighborSearchMethod(pcl::NeighborSearchMethod::DIRECT7);
  run_kitti(clouds, ndt_omp, "ndt_omp_direct7");

  ndt_omp.setNeighborSearchMethod(pcl::NeighborSearchMethod::DIRECT1);
  run_kitti(clouds, ndt_omp, "ndt_omp_direct1");

  ndt_omp.setNeighborSearchMethod(pcl::NeighborSearchMethod::KDTREE);
  run_kitti(clouds, ndt_omp, "ndt_omp_kdtree");

  ndt_omp.setNumThreads(1);
  ndt_omp.setNeighborSearchMethod(pcl::NeighborSearchMethod::DIRECT7);
  run_kitti(clouds, ndt_omp, "ndt_omp_direct7_st");

  ndt_omp.setNeighborSearchMethod(pcl::NeighborSearchMethod::DIRECT1);
  run_kitti(clouds, ndt_omp, "ndt_omp_direct1_st");

  ndt_omp.setNeighborSearchMethod(pcl::NeighborSearchMethod::KDTREE);
  run_kitti(clouds, ndt_omp, "ndt_omp_kdtree_st");

  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setTransformationEpsilon(translation_epsilon);
  ndt.setResolution(ndt_resolution);
  run_kitti(clouds, ndt, "ndt_original");

  std::cout << "done" << std::endl;

  return 0;
}
