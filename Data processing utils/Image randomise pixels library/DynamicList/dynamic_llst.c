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


// LINKED LIST FUNCTONS
Node* createLinkedList()
{
    /**
    This method creates the 1st empty node(root) in the list and then returns it.
    **/

    Node* n=(Node *)malloc(sizeof(Node));
    n->prev_node=(Node*)null;
    n->next_node=(Node*)null;
    n->index=0;
    return n;
}

int addItem(Node* root)
{
    /**
    This method takes the linked list and then adds a node(empty or not) at the end of the list.
    The new list size after operation is returned.
    **/
    Node* n=root;

    while(1)
    {
        if (n->next_node==(Node *)null)
        {
            Node* new_node=(Node *)malloc(sizeof(Node)); //the scope dies when you keave the method

            n->next_node=new_node;

            new_node->prev_node=n;
            new_node->next_node=(Node*)null;
            new_node->index=n->index+1;

            return new_node->index;
        }
        else
        {
            n=(n->next_node);
        }
    }

}

int popItem(Node* root)
{
    /**
    This method takes the linked list and then pops/removes the last element in the list.
    The new list size after operation is returned.
    **/

    Node* n=root;

    while(1)
    {
        if (n->next_node==(Node *)null)
        {
            int return_value=n->prev_node->index;

            n->prev_node->next_node=(Node *)null;
            free(n->p);
            free(n);
            return return_value;
        }
        else
        {
            n=(n->next_node);
        }
    }
}

int addDataToItem(Node* root,Person * data,int index)
{
    /**
    This methid takes the linked list and then adds the passed data at the index specified.

    If the index is found/data adding is successful , return 0
    If the index is not found/data not successfully added, return 1
    **/

    Node* n=root;

    while(1)
    {
        if (n->index == index)
        {
            n->p=data;
            return 0;
        }


        if (n->next_node==(Node *)null)
        return 1;

        n=(n->next_node);
    }
}


// PERSON CENTERED FUNCTIONS
Person* getItemData(Node* root,int index)
{
    /**
    This methods takes a linked list and then returns a data
    object at the index specified.

    Returns
    -------
    Person* : Person data at index.
    **/

    Node* n=root;

    while(1)
    {
        if (n->index == index)
        return n->p;

        if (n->next_node==(Node *)null)
        return (Person*)1;

        n=(n->next_node);
    }
}

void printPerson(Person * p)
{
    printf("       ||       \n");
    printf("       \\/       \n");
    printf("----------------\n");
    printf(" Name :%s\n",p->name);
    printf(" Age : %d\n",p->age);
    printf("----------------\n");
}

void printLinkedList(Node* root)
{
    /**
    Visually print the entire lined list on the terminal given the linked list object.
    **/

   Node* n=root;

    while(1)
    {
        printPerson(n->p);

        if (n->next_node==(Node *)null)
        return;

        n=n->next_node;
    }
}


// TEST MODULE
int main()
{
    /**
    Test the linked list and all the functions it provides.
    **/

    Node* linkedList=createLinkedList(); // #0
    addItem(linkedList);// #1
    addItem(linkedList);// #2

    Person * p1=(Person*)malloc(sizeof(Person));
    Person * p2=(Person*)malloc(sizeof(Person));
    Person * p3=(Person*)malloc(sizeof(Person));

    p1->name="Katlego"; //this is stored in the global read-only memory - data segment (not in the heap).You cannot edit this data.
    p1->age=25;

    p2->name="James";//this is stored in the global read-only memory - data segment (not in the heap).You cannot edit this data.
    p2->age=21;

    p3->name="Lucas";//this is stored in the global read-only memory - data segment (not in the heap).You cannot edit this data.
    p3->age=43;

    addDataToItem(linkedList,p1,0);
    addDataToItem(linkedList,p2,1);
    addDataToItem(linkedList,p3,2);

    popItem(linkedList);
    popItem(linkedList);


    printLinkedList(linkedList);


    return 0;

}

