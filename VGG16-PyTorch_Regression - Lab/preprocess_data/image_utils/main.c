#include <stdio.h>
#include <stdlib.h>

typedef char uint8_t;
typedef signed char int8_t;
typedef short int int16_t;
typedef short unsigned int uint16_t;

typedef struct {
    float L;
    float a;
    float b;
} Pixel;

//Prototypes
void mod_lab_range(float *flatted_img,int size);

//Functions
/*This is flattened image such that it's input size is Nx3*/
void mod_lab_range(float *flatted_img,int size){
    Pixel *p;
    float *IMG=flatted_img;

    for (int i=0;i<size;i++) {
        p=(Pixel*)((IMG+i*3));
        p->a+=127;
        p->b+=127;
    }
    return;
}

//Functions
/*int main()
{
    //Create dummt data
    float p1[3]={1,2,3};
    float p2[3]={4,5,6};
    float* arr[2]={p1,p2};

    //Conv dummy data to ptr
    float **arr_ptr=arr;

    //Run test function
    mod_lab_range(arr_ptr,2);

    //Validate results
    //printf("Function ran\n");
    //printf("%d",p1[2]);

    //test types
    //printf("%d",sizeof(int));

    return 0;
}*/
