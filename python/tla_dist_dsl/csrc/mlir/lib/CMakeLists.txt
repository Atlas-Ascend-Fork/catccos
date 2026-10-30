add_subdirectory(Passes)
add_subdirectory(Tools)

# Compatibility target for CATCCOS projects that linked the original flat
# compiler target.
add_library(TlaDistCompiler INTERFACE)
target_link_libraries(TlaDistCompiler INTERFACE TlaDistCompilePipeline TlaDistPasses)
