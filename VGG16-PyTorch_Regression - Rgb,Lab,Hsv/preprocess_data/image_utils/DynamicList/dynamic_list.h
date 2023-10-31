#include "../Headers/typedefs.h"
#include "../Headers/commons.h"

//--------------------------------

#ifndef DYNAMIC_LIST_H

typedef struct Node{
    struct Node* prev_node;
    struct Node* next_node;
    float data[9];
} Node;

typedef struct LinkedList{
    Node * root;
    Node * end;
    uint64_t length;
} LinkedList;

LinkedList* createLinkedList(Node * n);
Node* createEmptyNode();
Node* createDataNode(float * data,uint16_t length);
int32_t getItem(LinkedList* l,Node* result, uint32_t index);
float* getItemData(LinkedList* l, uint32_t index);
int32_t insertItem(LinkedList* l, Node* n, uint32_t index);
void appendItem(LinkedList* l,Node* n);
int32_t popItem(LinkedList* l,uint32_t index);

#endif // DYNAMIC_LIST_H
