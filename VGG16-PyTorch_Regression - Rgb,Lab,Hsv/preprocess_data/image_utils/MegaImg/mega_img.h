#include "../Headers/typedefs.h"
#include "../Headers/commons.h"

//--------------------------------

#ifndef MEGA_IMAGE_H

typedef struct Channels{
    float channel[9];
    boolean read;
} Channels;

typedef struct MegaImage{
    Channels data[224*224];
} MegaImage;

MegaImage* create_image_object();
boolean destroy_image_object(MegaImage* img);
Channels* get_random_pixel(MegaImage* img, uint32_t index);

#endif // MEGA_IMAGE_H
