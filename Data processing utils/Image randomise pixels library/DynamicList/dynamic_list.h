
#ifndef DYNAMIC_LIST_H

#define null -1
#define true 1
#define false 0

typedef unsigned short int boolean;
typedef cht uint8_t

typedef struct{
    char *data;
} String;

typedef struct{
    char *name;
    int *age;
} Person;

typedef struct Node{

struct Node* prev_node;
struct Node* next_node;
int index;
uint8_t* data;         //byte array
}Node;


Node* createLinkedList();
int addItem(Node* root);
int popItem(Node* root);
int addDataToItem(Node* root,Person * data,int index);
Person* getItemData(Node* root,int index);
void printPerson(Person * p);
void printLinkedList(Node* root);


#endif // DYNAMIC_LIST_H
