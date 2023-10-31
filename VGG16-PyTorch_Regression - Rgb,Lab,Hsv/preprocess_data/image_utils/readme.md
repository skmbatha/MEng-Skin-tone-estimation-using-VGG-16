# Use this command to compile to library
gcc main.c -shared -o image_utils.so

# To build the Library(image_utils_lib.so), run 'build_lib.bat'
The bat file will compile the main and dymanic_list C files accordingly,
thereafter it will aggregate the object files, and then create an output dynamic library.