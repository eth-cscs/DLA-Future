find_package(Doxygen REQUIRED)

set(DOXYGEN_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  @ONLY
)

add_custom_target(doc ALL
  COMMENT "Building Doxygen documentation"
  COMMAND Doxygen::doxygen
)
