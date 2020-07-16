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

add_executable(ndt_benchmark ndt_benchmark.cpp)
target_link_libraries(ndt_benchmark pcl_registration pcl_filters)