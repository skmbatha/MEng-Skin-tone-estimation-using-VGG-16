#include <stdio.h>
#include <stdlib.h>
#include "dynamic_list.h"


/**
Why use pointers and malloc for structures (especially if you want to mutate the structure)?

1.  You can create functions that mutate the structure (no scope limitations), scope becomes an issue when you define an
    object that is defined (without a reference) inside a function. Well, the point is, when you leave the function, the reference is lost
    even if you stored the pointer address(assigned at the time of existance). What actually happens is that, the data is wiped from memory.

2.  Structures who's size doesn't change/mutate can be declared as non-pointer based. This means you only change the values defined and the
    structure is accessible in different functions(within the same file). The functions can mutate the values but not structure. An instance
    of this could easily be a grouping of variables e.h the "debug_stack" strcture used in the HELGA code.

    To have access to the structure in other files(or anywhere in the application), the struct instance must be declared as "extern"!
**/

// Public:  LINKED LIST FUNCTONS
LinkedList* createLinkedList(Node * root)
{
    /**
    This method creates the 1st empty node(root) in the list and then returns it.
    **/

    LinkedList* n=(LinkedList *)malloc(sizeof(LinkedList));
    n->root=root;
    n->end=root;
    n->length=(uint64_t)1;
    return n;
}

Node* createEmptyNode()
{
    /**
    This method creates the 1st empty node(root) in the list and then returns it.
    **/

    Node* n=(Node *)malloc(sizeof(Node));
    n->prev_node=(Node*)null;
    n->next_node=(Node*)null;
    return n;
}

Node* createDataNode(float * data,uint16_t length)
{
    /**
    This method creates the 1st empty node(root) in the list and then returns it.
    **/

    //create & copy data : optimise for speed by usig v2
    /*float* d= (float*)malloc(sizeof(float)*length);
    for(int i=0;i<length;i++) {
        *(d+i)=*(data+i);
    }*/

    //create node and copy data
    Node* n=(Node *)malloc(sizeof(Node));
    n->prev_node=(Node*)null;
    n->next_node=(Node*)null;

    //V2 : Copy data to node
    for(int i=0;i<length;i++) {
        n->data[i]=data[i];
    }

    return n;
}

int32_t insertItem(LinkedList* l, Node* n, uint32_t index)
{
    /**This method will take ina Node object and insert it at the specified index on the list*/
    return 0;
}

void appendItem(LinkedList* l,Node* n)
{
    /**This method will append an item at the end of the list*/
    n->prev_node=l->end;
    l->end->next_node=n;
    l->end=n;
    l->length++;
}

int32_t getItem(LinkedList* l, Node* result, uint32_t index)
{
    /**This methods will return the requested Node object at the index location specified.
       The resultant value will be stored in the arg: result. The return value will either
       be 0(value found) or -1(null)*/

    if(index<l->length) {
        Node* current_node=l->root;
        for(int i=0; i<index; i++) {
            current_node=current_node->next_node;
        }
        result = current_node;
        return 0;
    } else {
        return null;
    }
}

float* getItemData(LinkedList* l, uint32_t index)
{
    /**This methods will return the requested Data from Node object at the index location specified.
       The return value will either be a float pointer at the data's location(in teh heap memory) or
       -1(null) ifthe index specified is out of range*/

    if(index<l->length) {
        Node* current_node=l->root;
        for(int i=0; i<index; i++) {
            current_node=current_node->next_node;
        }
        return current_node->data;
    } else {
        return (float*) null;
    }
}

int32_t popItem(LinkedList* l, uint32_t index)
{
    /**This methods will remove a Node object at the index specified
    If successful, the function will return 0 else null(-1)*/

    if(index<l->length) {
        Node* current_node=l->root;

        //Find indexed node
        for(int i=0;i<index;i++) {
            current_node=current_node->next_node;
        }

        //Connect prev'previs exists
        if(current_node->prev_node!=(Node *)null) {
            current_node->prev_node->next_node=current_node->next_node;
        }

        //Connect next's next node
        if(current_node->next_node!=(Node *)null) {
            current_node->next_node->prev_node=current_node->prev_node;
        }

        //if root is removed, add new root
        if(index==0) {
            l->root=current_node->next_node;
        }

        free(current_node);
        l->length--;

        return 0;
    } else {
        return null;
    }
}

