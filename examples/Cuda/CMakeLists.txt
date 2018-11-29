macro(EXAMPLE_CUDA EXAMPLE_CUDA_NAME)
    add_executable(${EXAMPLE_CUDA_NAME} "${EXAMPLE_CUDA_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CUDA_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXAMPLE_CUDA_NAME} PROPERTIES
            FOLDER "examples/Cuda/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
endmacro(EXAMPLE_CUDA)


EXAMPLE_CUDA(UniformSingleFrameMeshingCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(ScalableSingleFrameMeshingCuda ${CMAKE_PROJECT_NAME})

EXAMPLE_CUDA(UniformFusionSystemCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(ScalableFusionSystemCuda ${CMAKE_PROJECT_NAME})

EXAMPLE_CUDA(SequentialRGBDOdometryCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(ProfileRGBDOdometryCuda ${CMAKE_PROJECT_NAME})

EXAMPLE_CUDA(VisualizeSingleFramePointCloudCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(VisualizeSingleFrameMeshingCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(VisualizeTwoFrameRGBDOdometryCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(VisualizeTwoFrameRGBDOdometry ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(VisualizeTwoDepthFrames ${CMAKE_PROJECT_NAME})


