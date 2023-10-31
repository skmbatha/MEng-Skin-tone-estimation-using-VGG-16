#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Headers/typedefs.h"
#include "DynamicList/dynamic_list.h"
#include "MegaImg/mega_img.h"

typedef struct {
    float L;
    float a;
    float b;
} Pixel;

/******************************************
 *         CORE FUNCTIONS
*******************************************/

//Prototypes
void mod_lab_range(float *flatted_img,int size);
void randomise_pixels(int DIM, int SIZE,int REPEAT,float * R,float * G,float * B,float * H,float * S,float * V,float * L,float * a,float * b);

/*This is flattened image such that it's input size is Nx3*/
void mod_lab_range(float *flatted_img,int size){
    Pixel *p;
    float *IMG=flatted_img;

    for (int i=0;i<size;i++) {
        p=(Pixel*)((IMG+i*3));
        p->a+=110;
        p->b+=110;
    }
    return;
}

/*This function makes the pixels of the array passed in random*/
void randomise_pixels(int DIM, int SIZE,int REPEAT,float * R,float * G,float * B,float * H,float * S,float * V,float * L,float * a,float * b)
{
    float reg_1[9]={0,0,0,0,0,0,0,0,0};
    float reg_2[9]={0,0,0,0,0,0,0,0,0};
    float *img[9]={R,G,B,H,S,V,L,a,b};


    //Init random with seed
    srand((unsigned)time(0));

    //Initialise the temporal working registers
    /*for (int i;i<DIM;i++) {
        reg_1[i]=(float)0;
        reg_2[i]=(float)0;
    }*/

    //Run the pixel scattering algorithm
    int index=0;
    for (int r=0;r<REPEAT;r++) {

        //Process
        for (int i=0;i<SIZE;i++ ) {

            index=(int)(((float)(rand()/(float)RAND_MAX))*SIZE);

            //copy indexed value into reg_1
            for (int j=0;j<DIM;j++) {
                reg_1[j]=*(img[j]+index);
            }

            //copy contents of reg_2 to index
            for (int j=0;j<DIM;j++) {
                *(img[j]+index)=reg_2[j];
            }

            //copy contents of reg_1 to reg_2
            for (int j=0;j<DIM;j++) {
                reg_2[j]=reg_1[j];
            }
        }
    }
}

void randomise_pixels_2(int DIM, int SIZE,int REPEAT,float * R,float * G,float * B,float * H,float * S,float * V,float * L,float * a,float * b) 
{
    /**This implementation uses a linked list. Because of the # of mallocs/free involved in processing one image
     * is very large, therefore heavily impacting the performance of the algorithm.
    */

    //Init random with constant seed(randomisation pattern musn't change)
    srand(0);

    //Create linked list, init with
    float arr[9]={*R,*G,*B,*H,*S,*V,*L,*a,*b};
    LinkedList* l=createLinkedList(createDataNode((float*)arr,9));

    //Copy data to linked list
    for(int i=1;i<SIZE;i++) {
        float arr_temp[9]={*(R+i),*(G+i),*(B+i),*(H+i),*(S+i),*(V+i),*(L+i),*(a+i),*(b+i)};
        appendItem(l,createDataNode((float*)arr_temp,9)); 
    }

    //Randomly copy data at any index into original image
    int index=0;
    float *img[9]={R,G,B,H,S,V,L,a,b};
    for(int i=0;i<SIZE;i++) {
        index=(int)(((float)(rand()/(float)RAND_MAX))*(l->length-1));
        float* data=getItemData(l,index);
        for(int j=0;j<9;j++) { *(img[j]+i)=*(data+j); }
        if(popItem(l,index)!=0){ printf("Index not found\n");  }
    }
}

void randomise_pixels_3(int DIM, int SIZE,int SEED,float * R,float * G,float * B,float * H,float * S,float * V,float * L,float * a,float * b) 
{
    /** This function uses a different approch compared to randomise pixel 2. It uses a predefined image buffer that 
     * the image data can be initially copied into. After random indeces are requested from the copy.
     * When an index is read it's read varible in the Channels object is set to TRUE, if the index is 
     * requested again next time, the algorithm starts searching for the nearest unindexed pixel and returns it.
     * This mechanism makes sure that each request for a Pixel value using function "get_random_pixel" returns
     * a value. The execution speed of this functions as the pixels are read linearly dicreases from 100% -> 0%.
     * This measns the functio gets slower as the number of pixels->inf. 
    */

    //Init random with constant seed(randomisation pattern musn't change)
    srand(SEED);

    //Create image object
    MegaImage* img= create_image_object();

    //Initialise mega image
    float* arr[9]={R,G,B,H,S,V,L,a,b};
    for (int i=0;i<SIZE;i++) {
        for(int j=0;j<DIM;j++) {
            img->data[i].channel[j]=*(arr[j]+i);
            img->data[i].read=FALSE;
        }
    }

    //Randomise pixel
    int index=0;
    for(int i=0;i<SIZE;i++) {
        index=(int)(((float)(rand()/(float)RAND_MAX))*(SIZE));
        Channels* c=get_random_pixel(img,index);
        for(int j=0;j<DIM;j++) {
            switch(j) {
                /*case 0:*(arr[j]+i)=255; break;
                case 1:*(arr[j]+i)=0; break;
                case 2:*(arr[j]+i)=0; break;
                case 3:*(arr[j]+i)=0; break;
                case 4:*(arr[j]+i)=0; break;
                case 5:*(arr[j]+i)=0; break;
                case 6:*(arr[j]+i)=0; break;
                case 7:*(arr[j]+i)=0; break;
                case 8:*(arr[j]+i)=0; break;*/
                default:*(arr[j]+i)=c->channel[j]; 
            }
                
        }
    }

    //free image memory
    destroy_image_object(img);

}

/******************************************
 *         TEST FUNCTIONS/LIBS
*******************************************/
void test_linked_list();
void test_linked_list(){
    float arr1[4]={1,2,3,4};
    float arr2[5]={5,6,7,8,13};
    float arr3[2]={9,10};

    LinkedList* l=createLinkedList(createDataNode((float*)arr1,4)); //Create a linked list and initialise it with data
    appendItem(l,createDataNode((float*)arr2,5));                   //Append a node + data
    appendItem(l,createDataNode((float*)arr3,2));                   //Append a node + data

    printf("Linked List size %d\n",l->length);                      //Check the current Linked list size
    printf("pop item result %d\n",popItem(l,10));
    printf("Linked List size %d\n",l->length);
    printf("pop item result %d\n",popItem(l,2));
    printf("Linked List size %d\n",l->length);

    float* data=getItemData(l,1); // consider longjmp & setjmp
    if(data==(float*)null) {
        printf("No value received!");
    } else {
        printf("Value %f",data[2]);
    }
}
int main()
{
    test_linked_list();
    return 0;
}