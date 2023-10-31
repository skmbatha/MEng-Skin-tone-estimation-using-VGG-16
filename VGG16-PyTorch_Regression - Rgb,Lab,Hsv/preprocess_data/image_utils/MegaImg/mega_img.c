#include <stdio.h>
#include <stdlib.h>
#include "mega_img.h"

#define MIN_IMG_INDEX (int32_t)0
#define MAX_IMG_INDEX (int32_t)224*224


MegaImage* create_image_object(){
    /**/
    MegaImage* i = (MegaImage*)malloc(sizeof(MegaImage));
    return i;
}

boolean destroy_image_object(MegaImage* i){
    /**/
    free(i);
}

Channels* get_random_pixel(MegaImage* i, uint32_t index){
    /**/

    //Downwards search
    for (int32_t j=index;j>=MIN_IMG_INDEX;j--){
        if(i->data[j].read==FALSE){
            i->data[j].read=TRUE;
            return &(i->data[j]);
        }
    }

    //Upwards search
    for (int32_t j=index+1;j<MAX_IMG_INDEX;j++){
        if(i->data[j].read==FALSE){
            i->data[j].read=TRUE;
            return &(i->data[j]);
        }
    }

    
}

/*int main()
{
    //Test function
    for (int32_t j=1;j>=MIN_IMG_INDEX;j--){
        printf("Entered function @ j:%d\n",j);
    }
    printf("End of prog\n");
    return 0;
}*/