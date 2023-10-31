:: http://www.microhowto.info/howto/build_a_shared_library_using_gcc.html

gcc -c -fPIC -o output\main.o main.c
cd DynamicList & gcc -c -fPIC -o ..\output\dynamic_list.o dynamic_llst.c
cd ..\MegaImg & gcc -c -fPIC -o ..\output\mega_img.o mega_img.c
cd .. & gcc -shared -fPIC -o image_utils_lib.so output\main.o output\dynamic_list.o output\mega_img.o