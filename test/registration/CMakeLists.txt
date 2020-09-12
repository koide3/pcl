set(SUBSYS_NAME tests_registration)
set(SUBSYS_DESC "Point cloud library registration module unit tests")
PCL_SET_TEST_DEPENDENCIES(SUBSYS_DEPS registration)
set(OPT_DEPS io) # module does not depend on these

set(DEFAULT ON)
set(build TRUE)
PCL_SUBSYS_OPTION(build "${SUBSYS_NAME}" "${SUBSYS_DESC}" ${DEFAULT} "${REASON}")
PCL_SUBSYS_DEPEND(build "${SUBSYS_NAME}" DEPS ${SUBSYS_DEPS} OPT_DEPS ${OPT_DEPS})

if(NOT build)
  return()
endif()

PCL_ADD_TEST(ndt test_ndt
             FILES test_ndt.cpp
             LINK_WITH pcl_gtest pcl_io pcl_registration
             ARGUMENTS "${PCL_SOURCE_DIR}/test/bun0.pcd" "${PCL_SOURCE_DIR}/test/bun4.pcd")
