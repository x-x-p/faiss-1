file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../../)	# faiss root directory

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)
    message("${faiss_lib_gpu}")
    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} ${faiss_lib} gpufaiss)

    # Install
    install(TARGETS ${name} DESTINATION bin)
endforeach(source)
